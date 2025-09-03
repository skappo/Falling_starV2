import argparse
import cv2
import numpy as np
import time
import threading
import os
import subprocess
import logging
import json
import sys
import importlib
import shutil
import queue
from datetime import datetime
from collections import deque
from picamera2 import Picamera2
from libcamera import controls
from colorama import Fore, Style, init

# === CONTROLLO DIPENDENZE ===
REQUIRED_PACKAGES = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "picamera2": "picamera2",
    "ffmpeg": None,
    "colorama": "colorama"
}

missing = []

for module_name, pip_name in REQUIRED_PACKAGES.items():
    try:
        if module_name == "ffmpeg":
            if not shutil.which("ffmpeg"):
                raise ImportError
        else:
            importlib.import_module(module_name)
    except ImportError:
        missing.append(pip_name or module_name)

if missing:
    print("\n🚫 Dipendenze mancanti rilevate:")
    for pkg in missing:
        print(f"  - {pkg}")
    print("\n➡️  Puoi installarle con:")
    print(f"  pip install {' '.join([pkg for pkg in missing if pkg])}")
    print("🔧 Oppure su Raspberry Pi:\n  sudo apt install python3-opencv python3-numpy python3-picamera2 ffmpeg\n")
    sys.exit(1)

# === RILEVAMENTO MODELLO PI ===
def detect_pi_model():
    try:
        with open("/proc/device-tree/model", "r") as f:
            model_info = f.read().lower()
            if "raspberry pi 5" in model_info: return "pi5"
            elif "raspberry pi 4" in model_info: return "pi4"
    except: pass
    return "pi4"

pi_model = detect_pi_model()

# === FUNZIONE PER SALVARE LA CONFIGURAZIONE ===
CONFIG_FILE = "config.json"
def save_config(args):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(args, f, indent=4)
        print(f"[CONFIG] Configurazione aggiornata e salvata in {CONFIG_FILE}")
    except Exception as e:
        print(f"[ERROR] Could not save config to {CONFIG_FILE}: {e}")
         
# === ARGOMENTI CLI E CONFIGURAZIONE CENTRALE ===
parser = argparse.ArgumentParser(description="Rilevamento eventi luminosi e timelapse con Picamera2", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# --- Master defaults ---
master_defaults = {
    "size": "medium", "binning": True, "gain": 8.0, "codec": "h264", "output_dir": "output",
    "framerate_mode": "fixed", "framerate": 30 if pi_model == "pi5" else 20,
    # Meteor Finder settings
    "meteor_start_time": "22:00", "meteor_end_time": "05:00",
    "strategy": "contour", "record_duration": 10, "pre_event_seconds": 5,
    "min_brightness": 50, "min_area": 5,
    "diff_threshold": 25, "trigger_area": 4000, "learning_rate": 0.01,
    # Timelapse settings
    "timelapse_start_time": None, "timelapse_end_time": None,
    "timelapse_exposure": 15, "timelapse_interval": 20,
    "timelapse_gain": 8.0, "timelapse_color": False,
    "timelapse_to_video": False, "timelapse_video_fps": 24, "timelapse_cleanup_images": False,
    # Performance & Logging
    "no_log_events": False, "shutdown_time": None,
    "frame_queue_maxsize": 30, "output_queue_maxsize": 60,
    "snapshot_queue_maxsize": 10, "downscale_factor": 4
}
parser.set_defaults(**master_defaults)

# --- Define Arguments ---
# General
parser.add_argument("--size", choices=["small", "medium", "large"], help="Risoluzione per la cattura.")
parser.add_argument("--binning", action="store_true", help="Abilita il binning 2x2 per una maggiore sensibilità (solo per Meteor Finder).")
parser.add_argument("--gain", type=float, help="[Meteor] Gain analogico per il rilevamento meteore.")
parser.add_argument("--codec", choices=["avi", "mjpeg", "h264"], help="Codec video per gli eventi di meteore.")
parser.add_argument("--output_dir", type=str, help="Directory principale per salvare tutti i file.")
parser.add_argument("--framerate_mode", choices=["fixed", "dynamic"], help="Modalità di controllo del framerate per il rilevamento meteore.")
parser.add_argument("--framerate", type=int, help="Framerate target (usato solo in modalità 'fixed').")
# Meteor Finder
parser.add_argument("--meteor_start_time", type=str, help="Orario di inizio per il rilevamento meteore (HH:MM).")
parser.add_argument("--meteor_end_time", type=str, help="Orario di fine per il rilevamento meteore (HH:MM).")
parser.add_argument("--strategy", choices=["contour", "diff"], help="Strategia di rilevamento meteore.")
parser.add_argument("--record_duration", type=int, help="Durata della registrazione di un evento meteora.")
parser.add_argument("--pre_event_seconds", type=int, help="Secondi da bufferizzare prima di un evento meteora.")
parser.add_argument("--min_brightness", type=int, help="[Contour] Soglia di luminosità.")
parser.add_argument("--min_area", type=int, help="[Contour] Area minima del contorno.")
parser.add_argument("--diff_threshold", type=int, help="[Diff] Soglia di differenza tra pixel.")
parser.add_argument("--trigger_area", type=int, help="[Diff] Numero di pixel cambiati per attivare.")
parser.add_argument("--learning_rate", type=float, help="[Diff] Tasso di adattamento del reference frame.")
# Timelapse
parser.add_argument("--timelapse_start_time", type=str, help="Orario di inizio per il timelapse (HH:MM).")
parser.add_argument("--timelapse_end_time", type=str, help="Orario di fine per il timelapse (HH:MM).")
parser.add_argument("--timelapse_exposure", type=int, help="[Timelapse] Tempo di esposizione in secondi per ogni scatto.")
parser.add_argument("--timelapse_interval", type=int, help="[Timelapse] Intervallo in secondi tra uno scatto e l'altro.")
parser.add_argument("--timelapse_gain", type=float, help="[Timelapse] Gain analogico (ISO) per gli scatti del timelapse.")
parser.add_argument("--timelapse_color", action="store_true", help="[Timelapse] Salva le immagini del timelapse a colori.")
parser.add_argument("--timelapse_to_video", action="store_true", help="[Timelapse] Crea un video finale dalle immagini JPG al termine.")
parser.add_argument("--timelapse_video_fps", type=int, help="[Timelapse] FPS del video finale creato dalle immagini.")
parser.add_argument("--timelapse_cleanup_images", action="store_true", help="[Timelapse] Cancella le immagini JPG originali dopo aver creato il video.")
# Performance & Logging
parser.add_argument("--no_log_events", action="store_true", help="Disabilita il logging.")
parser.add_argument("--frame_queue_maxsize", type=int, help="Dimensione massima della coda di analisi meteore.")
parser.add_argument("--output_queue_maxsize", type=int, help="Dimensione massima della coda di scrittura video meteore.")
parser.add_argument("--snapshot_queue_maxsize", type=int, help="Dimensione massima della coda di scrittura snapshot meteore.")
parser.add_argument("--downscale_factor", type=int, help="Fattore di ridimensionamento per lo stream lores (rilevamento meteore).")
parser.add_argument("--shutdown_time", type=str, help="Orario per lo spegnimento automatico del sistema (HH:MM). Richiede privilegi sudo.")

# Carica il file config.json se esiste. I suoi valori sovrascrivono i master_defaults.
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        try:
            config_from_file = json.load(f)
            # The file's content now becomes our new set of defaults
            master_defaults.update(config_from_file)
            print(f"[CONFIG] Caricata configurazione da {CONFIG_FILE}")
        except Exception as e:
            print(f"[WARNING] Impossibile analizzare {CONFIG_FILE}: {e}")
else:
    print(f"[CONFIG] File di configurazione non trovato. Creazione di {CONFIG_FILE} con i valori predefiniti.")
    save_config(master_defaults)

# Set these combined values as the defaults for argparse
parser.set_defaults(**master_defaults)

# Se ci sono argomenti dalla riga di comando sovrascrivono TUTTO il resto.
args = parser.parse_args()

# --- Save back to config.json ONLY if CLI args were used ---
# We rebuild the final config dictionary from the parsed args
final_config = vars(args)

# Compare the final config with the defaults loaded at the start.
# If they are different, it means the user provided CLI args that should be saved.
if final_config != master_defaults:
    print("[CONFIG] Rilevati argomenti da riga di comando. Aggiornamento del file di configurazione...")
    save_config(final_config)

# Imposta la strategia di riconoscimento
STRATEGY = args.strategy

# === LOGGING (with Color) ===
init(autoreset=True)

class ColorFormatter(logging.Formatter):
    COLORS = {
        "INFO": Style.BRIGHT + Fore.WHITE,
        "WARNING": Style.BRIGHT + Fore.YELLOW,
        "ERROR": Style.BRIGHT + Fore.RED,
        "DEBUG": Style.DIM + Fore.CYAN,
    }

    CUSTOM = {
        "[EVENT]": Fore.GREEN + Style.BRIGHT,
        "[METEOR_FINDER]": Fore.BLUE + Style.BRIGHT,
        "[TIMELAPSE]": Fore.CYAN + Style.BRIGHT,
        "[POST-PROCESSING]": Fore.MAGENTA + Style.BRIGHT,
        "[SNAPSHOT]": Fore.YELLOW,
        "[MANAGER]": Fore.WHITE + Style.BRIGHT,
        "[MONITOR]": Fore.LIGHTBLACK_EX,
        "[SHUTDOWN]": Fore.LIGHTYELLOW_EX + Style.BRIGHT,
    }

    def format(self, record):
        msg = super().format(record)
        # Apply log level color
        color = self.COLORS.get(record.levelname, "")
        msg = f"{color}{msg}{Style.RESET_ALL}"
        # Apply tag colors
        for tag, tag_color in self.CUSTOM.items():
            if tag in msg:
                msg = msg.replace(tag, f"{tag_color}{tag}{Style.RESET_ALL}")
        return msg

# Root logger reset
logging.getLogger().handlers.clear()
logging.getLogger().setLevel(logging.INFO)

# Console handler with colors
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColorFormatter("%(asctime)s [%(levelname)s] %(message)s"))

# File handler without colors
file_handler = logging.FileHandler("detection_log.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logging.getLogger().addHandler(console_handler)
logging.getLogger().addHandler(file_handler)

if args.no_log_events:
    logging.disable(logging.CRITICAL)

# === PARAMETRI DERIVATI ===
RESOLUTIONS = {"small": (640, 480), "medium": (1280, 720), "large": (1920, 1080)}

# === ESPOSIZIONE ===
METEOR_EXPOSURE_LIMITS = (50000, 1000000)

# === NUOVA FUNZIONE: CREAZIONE VIDEO TIMELAPSE ===
def create_timelapse_video(image_folder, output_filename, fps, cleanup=False):
    """
    Crea un video MP4 da una sequenza di immagini JPG usando ffmpeg.
    """
    logging.info(f"[POST-PROCESSING] Avvio creazione video timelapse da {image_folder}")
    
    # Trova e ordina tutte le immagini JPG nella cartella
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])
    if not images:
        logging.warning("[POST-PROCESSING] Nessuna immagine trovata per creare il video timelapse.")
        return

    # Crea un file di testo temporaneo che elenca i file di immagine per ffmpeg
    list_filename = os.path.join(image_folder, "imagelist.txt")
    with open(list_filename, 'w') as f:
        for img in images:
            f.write(f"file '{img}'\n")

    # Costruisce il comando ffmpeg
    # Questo comando è robusto e produce video di alta qualità
    cmd = [
        "ffmpeg",
        "-y",                             # Sovrascrive il file di output se esiste
        "-r", str(fps),                   # FPS del video in uscita
        "-f", "concat",                   # Usa il formato concatenatore
        "-safe", "0",
        "-i", list_filename,              # File di input con l'elenco delle immagini
        "-c:v", "libx264",                # Codec video H.264
        "-preset", "slow",                # Preset di qualità (slow = migliore compressione)
        "-crf", "18",                     # Fattore di qualità (più basso è meglio, 18 è quasi lossless)
        "-pix_fmt", "yuv420p",            # Formato pixel per la massima compatibilità
        output_filename
    ]

    try:
        # Esegui il comando ffmpeg
        subprocess.run(cmd, cwd=image_folder, check=True, capture_output=True, text=True)
        logging.info(f"[POST-PROCESSING] Video timelapse creato con successo: {output_filename}")

        # Pulisci i file temporanei e le immagini originali se richiesto
        os.remove(list_filename)
        if cleanup:
            logging.info("[POST-PROCESSING] Pulizia delle immagini JPG originali...")
            for img in images:
                os.remove(os.path.join(image_folder, img))
            logging.info("[POST-PROCESSING] Pulizia completata.")

    except subprocess.CalledProcessError as e:
        logging.error(f"[POST-PROCESSING] Errore durante la creazione del video timelapse con ffmpeg.")
        logging.error(f"  Comando: {' '.join(cmd)}")
        logging.error(f"  Output di ffmpeg:\n{e.stderr}")
    except Exception as e:
        logging.error(f"[POST-PROCESSING] Errore imprevisto: {e}")

# === FUNZIONI DI SCHEDULING ===
def is_time_in_interval(start_str, end_str):
    if not start_str or not end_str:
        return False
    now = datetime.now().time()
    start = datetime.strptime(start_str, "%H:%M").time()
    end = datetime.strptime(end_str, "%H:%M").time()
    if start < end:
        return start <= now < end
    else: # Handles overnight interval
        return now >= start or now < end

# === RILEVAMENTO CAMERA CON AUTOFOCUS ===
def is_autofocus_camera(picam2_object):
    # Camera Info if v3 disable autofocus
    try:
        model = picam2_object.camera_properties.get('Model', 'unknown').lower()
        if 'imx708' in model:
            logging.info(f"Rilevato sensore IMX708 (Camera Module 3). Verranno applicati i controlli di fuoco manuale per il timelapse.")
            return True
    except Exception as e:
        logging.error(f"Impossibile determinare il modello della camera: {e}")
    return False

# === INIZIALIZZAZIONE CAMERA (Iniziale) ===
os.makedirs(args.output_dir, exist_ok=True)
picam2 = Picamera2()
HAS_AUTOFOCUS = is_autofocus_camera(picam2) # Rileva una volta all'avvio
width, height = RESOLUTIONS[args.size]

# === SUPPORTO H264 PER RASPBERRY PI ===
def start_ffmpeg_writer(filename, width, height, framerate):
    # Avvia un processo ffmpeg che accetta dati raw in grayscale e li codifica in H.264 su un raspberry utilizzando h264_v4l2m2m. 
    cmd = [
        "ffmpeg", "-y", 
        "-f", "rawvideo", 
        "-pix_fmt", "gray",
        "-s", f"{width}x{height}",
        "-r", str(framerate), 
        "-i", "-", 
        "-vf", "format=yuv420p",
        "-an",
        "-vcodec", "h264_v4l2m2m", # <-- USA L'ENCODER HARDWARE del Raspberry Pi
        "-preset", "ultrafast", 
        "-crf", "23", 
        filename
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)

# === CODEC ===
# Determina l'estensione del file e il FourCC per i codec non-H264.
if args.codec == "avi":
    # fourcc = cv2.VideoWriter_fourcc(*'IYUV') # Old
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # More compatible
    extension = "avi"
elif args.codec == "mjpeg":
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    extension = "avi"
elif args.codec == "h264":
    fourcc = None
    extension = "mp4"
else:
    raise ValueError("Codec non supportato.")

# === CODE E VAR GLOBALI ===
# --- Queues ---
frame_queue = queue.Queue(maxsize=args.frame_queue_maxsize)
stack_queue = queue.Queue(maxsize=5) # Piccola, solo per i frame da aggiungere
output_queue = queue.Queue(maxsize=args.output_queue_maxsize)
snapshot_queue = queue.Queue(maxsize=args.snapshot_queue_maxsize)
timelapse_writer_queue = queue.Queue(maxsize=10)

# --- Shared State Variables ---
#pre_event_buffer = deque(maxlen=int(args.pre_event_seconds * args.framerate))
reference_frame = None
recording_event = threading.Event()
out, ffmpeg_proc = None, None
running = True
current_state = "IDLE"
last_event_time = 0
record_start_time = 0
frames_captured = 0
frames_processed = 0
frames_written = 0
events_triggered = 0
lock_perf = threading.Lock()


def update_perf_counter(counter_name, increment=1):
    # Funzione thread-safe per aggiornare i contatori di performance.
    global frames_captured, frames_processed, frames_written, events_triggered
    with lock_perf:
        if counter_name == "captured": frames_captured += increment
        elif counter_name == "processed": frames_processed += increment
        elif counter_name == "written": frames_written += increment
        elif counter_name == "events": events_triggered += increment

# === THREADS ===

# --- Universal Threads ---
def capture_thread_meteor(state_event, width, height):
    """
    Cattura frame ad alto framerate per il rilevamento meteore in modo robusto.
    Usa il pattern capture_request per garantire la corretta gestione dei dati.
    """
    # Pre-calcola le dimensioni dello stream a bassa risoluzione per lo slicing
    lores_w = width // args.downscale_factor
    lores_h = height // args.downscale_factor
    
    while running and state_event.is_set():
        request = None # Inizializza a None per sicurezza
        try:
            # --- THIS IS THE DEFINITIVE FIX ---
            # 1. Cattura la richiesta per entrambi gli stream.
            request = picam2.capture_request()

            # 2. Estrae i dati in array NumPy garantiti.
            main_frame_yuv = request.make_array("main")
            lores_frame_yuv = request.make_array("lores")
           
            # Converte in grayscale QUI, una sola volta.
            main_frame_gray = main_frame_yuv[:height, :width]
            lores_frame_gray = lores_frame_yuv[:lores_h, :lores_w]

            # 3. Ora che abbiamo array NumPy, li mettiamo nella coda.
            #    La conversione in grayscale avverrà nel processing_thread.
            frame_queue.put_nowait((main_frame_gray, lores_frame_gray))
            update_perf_counter("captured")

        except queue.Full:
            pass # Comportamento previsto
        except Exception as e:
            logging.error(f"[CAPTURE] Errore imprevisto durante la cattura del frame: {e}")
        finally:
            # 4. Rilascia sempre la richiesta per liberare i buffer.
            if request:
                request.release()

def monitor_thread():
    """Stampa aggiornamenti periodici sullo stato e contatori persistenti."""
    global frames_captured, frames_processed, frames_written, events_triggered, lock_perf
    INTERVAL = 15 # Aumentiamo l'intervallo per rendere i totali più significativi

    # Variabili per calcolare i tassi (FPS) tra un intervallo e l'altro
    last_fc, last_fp, last_fw, last_ev = 0, 0, 0, 0

    while running:
        time.sleep(INTERVAL)

        # Se lo script è inattivo, non stampare nulla e attendi il prossimo ciclo.
        if current_state == "IDLE":
            logging.info(f"[MONITOR] Stato @ {datetime.now().strftime('%H:%M:%S')}: IDLE. In attesa del prossimo task programmato.", extra={'state': 'IDLE'})
            continue
        
        # Legge i valori correnti dei contatori globali in modo thread-safe
        with lock_perf:
            current_fc = frames_captured
            current_fp = frames_processed
            current_fw = frames_written
            current_ev = events_triggered

        # --- Calcolo dei Tassi (Eventi per Intervallo) ---
        # Calcola quanti eventi si sono verificati dall'ultima stampa
        fc_interval = current_fc - last_fc
        fp_interval = current_fp - last_fp
        fw_interval = current_fw - last_fw
        
        # Aggiorna i valori "last" per il prossimo ciclo
        last_fc, last_fp, last_fw, last_ev = current_fc, current_fp, current_fw, current_ev

        # Calcola gli FPS basati solo sull'attività nell'ultimo intervallo
        fps_cap = fc_interval / INTERVAL
        fps_proc = fp_interval / INTERVAL
        fps_write = fw_interval / INTERVAL
        
        # --- Preparazione dell'Output ---
        status = "SÌ" if 'recording_event' in globals() and recording_event.is_set() else "NO"
        q_sizes = f"[{frame_queue.qsize()}/{output_queue.qsize()}/{snapshot_queue.qsize()}/{timelapse_writer_queue.qsize()}]"
        
        # --- Color helpers ---
        def color_state(state):
            mapping = {
                "METEOR_FINDER": Fore.BLUE + state + Style.RESET_ALL,
                "TIMELAPSE": Fore.CYAN + state + Style.RESET_ALL,
                "POST_PROCESSING": Fore.MAGENTA + state + Style.RESET_ALL,
                "IDLE": Fore.LIGHTBLACK_EX + state + Style.RESET_ALL,
            }
            return mapping.get(state, state)

        def color_queue(current, maximum):
            fill = current / maximum if maximum > 0 else 0
            if fill < 0.5: return Fore.GREEN + f"{current}" + Style.RESET_ALL
            elif fill < 0.8: return Fore.YELLOW + f"{current}" + Style.RESET_ALL
            else: return Fore.RED + f"{current}" + Style.RESET_ALL

        # Build colored queue strings
        q_frame = color_queue(frame_queue.qsize(), args.frame_queue_maxsize)
        q_out = color_queue(output_queue.qsize(), args.output_queue_maxsize)
        q_snap = color_queue(snapshot_queue.qsize(), args.snapshot_queue_maxsize)
        q_tl = color_queue(timelapse_writer_queue.qsize(), 10)

        # --- Output (same structure as before, but with colors) ---
        logging.info(f"[MONITOR] Stato @ {datetime.now().strftime('%H:%M:%S')}: {color_state(current_state)}")
        # Stampa i totali incrementali
        logging.info(f"  Totali Sessione -> Frame Acquisiti: {current_fc} | Frame Processati: {current_fp} | Frame Scritti: {current_fw}")
        logging.info(f"  Totali Sessione -> Eventi/Scatti Rilevati: {current_ev}")
        # Stampa i tassi attuali
        logging.info(f"  Tasso Attuale -> FPS (Acq/Proc/Scrit): {fps_cap:.1f}/{fps_proc:.1f}/{fps_write:.1f}")
        logging.info(f"  Stato -> Registrazione Attiva: {status}, Code [Meteora/Video/Snap/Timelapse]: "f"[{q_frame}/{q_out}/{q_snap}/{q_tl}]")

# --- Meteor Finder Threads ---
def processing_thread(state_event, effective_framerate, pre_event_buffer, width, height): 
    """
    Preleva i frame, esegue il rilevamento e gestisce la registrazione in modo
    robusto e thread-safe.
    """
    global reference_frame, last_event_time, out, ffmpeg_proc, record_start_time
    
    scaled_trigger_area = args.trigger_area / (args.downscale_factor**2)
    is_shutting_down = False
    
    while running and state_event.is_set():
        
        # --- Blocco di Arresto Prioritario ---
        # Questo blocco ha la priorità su tutto il resto. Se dobbiamo arrestare una
        # registrazione, ci concentriamo solo su quello.
        if is_shutting_down:
            try:
                output_queue.put(None, block=False) # Invia il segnale di stop
                logging.info("[PROCESS] Segnale di stop inviato con successo al writer.")
                
                # Resetta lo stato solo DOPO aver inviato il segnale con successo
                is_shutting_down = False
                pre_event_buffer.clear()
                recording_event.clear()
                
            except queue.Full:
                # La coda è ancora piena, il writer è indietro. Aspettiamo e riproviamo.
                time.sleep(0.5)
            
            # Torna all'inizio del loop per riprovare a inviare il segnale
            # o per uscire se lo stato del thread è cambiato.
            continue

        # --- Acquisizione Frame ---
        try:
            # Usa get_nowait per un loop reattivo che non si blocca.
            full_res_frame, detection_frame = frame_queue.get_nowait()
            update_perf_counter("processed")
        except queue.Empty:
            # È normale che la coda sia vuota, facciamo una breve pausa.
            time.sleep(0.01)
            continue
        
        # --- Logica di Rilevamento ---
        # Lavora sempre su frame in scala di grigi.
        blurred = cv2.medianBlur(detection_frame, 3)
        motion_detected = False

        if STRATEGY == "diff":
            if reference_frame is None:
                reference_frame = blurred.copy().astype("float")
                continue
            frame_diff = cv2.absdiff(blurred, cv2.convertScaleAbs(reference_frame))
            _, thresh = cv2.threshold(frame_diff, args.diff_threshold, 255, cv2.THRESH_BINARY)
            changed_area = cv2.countNonZero(thresh)
            motion_detected = changed_area > scaled_trigger_area
            cv2.accumulateWeighted(blurred, reference_frame, args.learning_rate)
        else: # contour strategy
            _, thresh = cv2.threshold(blurred, args.min_brightness, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_detected = any(cv2.contourArea(c) > args.min_area for c in contours)

        # --- Gestione Buffer e Registrazione ---
        pre_event_buffer.append(full_res_frame.copy())
        current_time = time.time()

        # Caso 1: Rilevato un nuovo evento
        if motion_detected and not recording_event.is_set() and (current_time - last_event_time > 2):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(args.output_dir, f"evento_{timestamp}.{extension}")
            snapshot_path = os.path.join(args.output_dir, f"snapshot_{timestamp}.jpg")
            
            try:
                snapshot_queue.put((snapshot_path, full_res_frame.copy()), block=False)
            except queue.Full:
                logging.warning("[PROCESS] Coda snapshot piena, snapshot scartato.")            
            
            if args.codec == "h264":
                ffmpeg_proc = start_ffmpeg_writer(filename, width, height, effective_framerate)
            else:
                out = cv2.VideoWriter(filename, fourcc, effective_framerate, (width, height), isColor=False)
            
            for f in pre_event_buffer:
                try:
                    output_queue.put(f, timeout=0.1)
                except queue.Full:
                    logging.warning("[PROCESS] Coda di output piena durante la scrittura del pre-evento.")

            recording_event.set()
            record_start_time = current_time
            last_event_time = current_time
            update_perf_counter("events")
            logging.info(f"[EVENT] Evento rilevato: registrazione iniziata {filename}")

        # Caso 2: La registrazione è già in corso
        elif recording_event.is_set():
            try:
                output_queue.put(full_res_frame, timeout=1)
            except queue.Full:
                logging.warning("[PROCESS] Coda di output piena, frame perso")

            # Controlla se è ora di terminare la registrazione
            if time.time() - record_start_time > args.record_duration:
                logging.info("[PROCESS] Durata registrazione terminata. Inizio procedura di arresto...")
                is_shutting_down = True

    # --- Uscita dal Loop Principale ---
    # Se il thread deve terminare (es. per cambio di modalità) mentre una registrazione
    # stava per essere arrestata, esegui un ultimo tentativo di inviare il segnale di stop.
    if is_shutting_down:
        try:
            output_queue.put(None, timeout=0.5)
            logging.info("[PROCESS] Segnale di stop finale inviato prima della terminazione.")
        except queue.Full:
            logging.error("[PROCESS] Impossibile inviare il segnale di stop finale. Il video potrebbe non essere finalizzato.")

def writer_thread(state_event):
    # Gestisce la scrittura dei frame video su disco.
    global out, ffmpeg_proc # Still needs access to the global file handles

    # The main loop checks for both the global running flag and the mode-specific event
    while running and state_event.is_set():
        try:
            # Attende un oggetto dalla coda. Potrebbe essere un frame o il segnale di stop (None).
            item = output_queue.get_nowait()

            # --- Logica di Chiusura Coordinata ---
            # Controlla se l'oggetto ricevuto è il segnale di "Fine Stream".
            if item is None:
                logging.info("[WRITE] Segnale di stop ricevuto. Finalizzazione del video in corso...")
                
                # Chiude in modo sicuro il processo ffmpeg, se attivo
                if ffmpeg_proc:
                    if ffmpeg_proc.stdin:
                        try:
                            ffmpeg_proc.stdin.close()
                        except BrokenPipeError:
                            # Questo può accadere se il processo è terminato in anticipo, è sicuro ignorarlo.
                            logging.warning("[WRITE] stdin di ffmpeg già chiuso.")
                            pass
                    ffmpeg_proc.wait() # Attende che il processo ffmpeg termini completamente
                    ffmpeg_proc = None # Resetta la variabile globale
                
                # Chiude in modo sicuro l'oggetto VideoWriter di OpenCV, se attivo
                if out:
                    out.release()
                    out = None # Resetta la variabile globale
                
                logging.info("[WRITE] Video finalizzato e chiuso correttamente.")
                
                # Continua al prossimo ciclo per attendere un nuovo task (non esce dal loop)
                continue

            # --- Logica di Scrittura del Frame ---
            # Se l'oggetto non è None, allora è un frame da scrivere.
            # Scriviamo solo se uno dei writer è effettivamente attivo.
            if out or ffmpeg_proc:
                try:
                    if ffmpeg_proc:
                        ffmpeg_proc.stdin.write(item.tobytes())
                    elif out:
                        # Converte al volo in grayscale per i codec non-H264
#                        frame_gray = item[:height, :width]
                        out.write(item)
                    update_perf_counter("written") # Se vuoi ripristinare il contatore
                except Exception as e:
                    # Gestisce errori che potrebbero verificarsi durante la scrittura
                    logging.error(f"[WRITE] Errore imprevisto durante la scrittura del frame: {e}")
                    # In caso di errore grave, è meglio resettare lo stato di registrazione
                    # per evitare un ciclo di errori.
                    if ffmpeg_proc and ffmpeg_proc.stdin:
                        try:
                            ffmpeg_proc.stdin.close()
                        except BrokenPipeError:
                            pass
                        ffmpeg_proc.wait()
                        ffmpeg_proc = None
                        if out:
                            out.release()
                            out = None
                    recording_event.clear()
        except queue.Empty:
            # È normale che la coda sia vuota, continua semplicemente ad attendere.
            time.sleep(0.01)
            continue
        
    logging.info("[WRITER] Thread di scrittura terminato.")

def snapshot_writer_thread(state_event):
    # Gestisce la scrittura degli snapshot su disco.
    while running and state_event.is_set():
        try:
            snapshot_path, frame_to_save = snapshot_queue.get_nowait()
            start = time.time()
            cv2.imwrite(snapshot_path, frame_to_save)
            duration = time.time() - start
            logging.info(f"[SNAPSHOT] Snapshot salvato in {snapshot_path} ({duration:.3f}s)")
        except queue.Empty:
            time.sleep(0.01)
            continue
        except Exception as e:
            logging.error(f"[SNAPSHOT] Errore nel salvataggio snapshot: {e}")

# --- Timelapse Threads ---
def timelapse_capture_thread(state_event, width, height):
    """Cattura scatti a lunga esposizione a intervalli regolari."""
    logging.info("[TIMELAPSE] Thread di cattura avviato.")
    last_capture_time = time.time() - args.timelapse_interval # Per scattare subito la prima foto

    while running and state_event.is_set():
        if time.time() - last_capture_time >= args.timelapse_interval:
            last_capture_time = time.time()
#            request = None
            try:
                wait_timeout = args.timelapse_exposure + 5
                logging.info(f"[TIMELAPSE] In attesa del completamento del job (timeout: {wait_timeout}s)...")

                # picam2.wait() attende il completamento di un job e restituisce la richiesta completata.
                request = picam2.wait(job, timeout=wait_timeout * 1000) # timeout è in millisecondi

                if request is None:
                    raise RuntimeError("La richiesta di cattura del timelapse è scaduta (il job non è stato completato in tempo).")
                    
                # Let Picamera2 build the proper NumPy array from the request
                captured_image = request.make_array('main')
                
                if args.timelapse_color:
                    final_frame = cv2.cvtColor(captured_image, cv2.COLOR_RGB2BGR)
                else:
                    final_frame = cv2.cvtColor(captured_image, cv2.COLOR_YUV2GRAY_I420)
                
                timelapse_writer_queue.put(final_frame)
                logging.info("[TIMELAPSE] Immagine catturata e inviata per il salvataggio.")
            except Exception as e:
                logging.error(f"[TIMELAPSE] Errore durante la cattura: {e}")
            finally:
                # This block will ALWAYS execute
                if 'request' in locals() and request:
                    request.release()
                    
        time.sleep(1) # Breve pausa per ridurre il carico della CPU
    logging.info("[TIMELAPSE] Thread di cattura terminato.")

def timelapse_writer_thread(state_event):
    """Salva su disco le immagini del timelapse."""
    timelapse_dir = os.path.join(args.output_dir, "timelapse")
    os.makedirs(timelapse_dir, exist_ok=True)
    logging.info(f"[TIMELAPSE] Thread di scrittura avviato. Salvataggio in: {timelapse_dir}")
    
    while running and state_event.is_set():
        try:
            frame = timelapse_writer_queue.get_nowait()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(timelapse_dir, f"tl_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            update_perf_counter("events")
        except queue.Empty:
            time.sleep(0.01)
            continue
    logging.info("[TIMELAPSE] Thread di scrittura terminato.")

# === MAIN: The State Machine Manager ===
if __name__ == "__main__":
    active_threads = []
    state_events = {'meteor_finder': threading.Event(), 'timelapse': threading.Event()}
    
    width, height = RESOLUTIONS[args.size]
    
    # --- Shutdown Time Setup ---
    shutdown_time_obj = None
    if args.shutdown_time:
        try:
            shutdown_time_obj = datetime.strptime(args.shutdown_time, "%H:%M").time()
            logging.info(f"[MANAGER] Spegnimento del sistema programmato per le {args.shutdown_time}")
        except ValueError:
            logging.error(f"[MANAGER] Formato dell'orario di spegnimento non valido: {args.shutdown_time}. La funzione è disabilitata.")
    
    shutdown_initiated = False
    shutdown_for_system = False

    # Avvia i thread universali
    logging.info("[MAIN] Avvio dei thread universali (Monitor)...")
    monitor_t = threading.Thread(target=monitor_thread, daemon=True, name="Monitor")
    monitor_t.start()
    
    try:
        while running:
            # 1. Determina lo stato desiderato in base alla pianificazione
            desired_state = "IDLE"
            is_meteor_time = is_time_in_interval(args.meteor_start_time, args.meteor_end_time)
            is_timelapse_time = is_time_in_interval(args.timelapse_start_time, args.timelapse_end_time)

            if is_meteor_time:
                desired_state = "METEOR_FINDER"
            elif is_timelapse_time:
                desired_state = "TIMELAPSE"
            elif current_state == "TIMELAPSE" and not is_timelapse_time and args.timelapse_to_video:
                desired_state = "POST_PROCESSING"

            # 2. Se lo stato deve cambiare, esegui la transizione
            if desired_state != current_state:
                logging.info(f"[MANAGER] Transizione di stato: {current_state} -> {desired_state}")

                # --- Fase di SHUTDOWN ---
                if current_state != "IDLE":
                    logging.info(f"[MANAGER] Arresto della modalità {current_state}...")

                    # 1. Signal all threads in the current mode to stop their loops.
                    state_events[current_state.lower()].clear()
                    
                    # 2. Wait for them to finish gracefully.
                    #    The writer thread will finish its current file thanks to the sentinel.
                    for t in active_threads:
                        t.join(timeout=10) # Give them a generous timeout to finish I/O

                    # 3. Check for any that did not terminate.
                    alive_threads = [t for t in active_threads if t.is_alive()]
                    if alive_threads:
                        logging.warning(f"[MANAGER] I seguenti thread non sono terminati: {[t.name for t in alive_threads]}")
                    
                    active_threads.clear()
                    
                    # 4. Only stop the camera after all threads that use it are confirmed down.
                    if picam2.started:                        
                        picam2.stop()
                        
                    logging.info(f"[MANAGER] Modalità {current_state} arrestata.")
                
                # --- Fase di STARTUP ---
                current_state = desired_state
                if current_state == "METEOR_FINDER":
                    logging.info("[MANAGER] Riconfigurazione della camera per METEOR_FINDER...")
                    lores_w = width // args.downscale_factor
                    lores_h = height // args.downscale_factor
                    
                    # Costruisce i controlli di base
                    meteor_controls = {"FrameDurationLimits": METEOR_EXPOSURE_LIMITS, "AnalogueGain": args.gain}
                    
                    # Aggiunge il controllo del framerate solo se la modalità è 'fixed'
                    if args.framerate_mode == 'fixed':
                        meteor_controls["FrameRate"] = args.framerate
                        
                    video_config = picam2.create_video_configuration(
                        main={"size": (width, height), "format": "YUV420"},
                        lores={"size": (lores_w, lores_h), "format": "YUV420"},
                        controls=meteor_controls
                    )
                    picam2.configure(video_config)
                    picam2.start()

                    if args.framerate_mode == 'dynamic':
                        metadata = picam2.capture_metadata()
                        effective_framerate = metadata.get("FrameRate")
                        logging.info(f"[CAMERA] Framerate dinamico rilevato: {effective_framerate:.2f} fps")
                    else: # 'fixed'
                        effective_framerate = args.framerate
                        logging.info(f"[CAMERA] Framerate fisso impostato a: {effective_framerate} fps")

                    logging.info(f"[MANAGER] Creazione del pre-event buffer per {args.pre_event_seconds} secondi ({int(args.pre_event_seconds * effective_framerate)} frames).")                   
                    pre_event_buffer = deque(maxlen=int(args.pre_event_seconds * effective_framerate))

                    active_threads = [
                        threading.Thread(target=capture_thread_meteor, args=(state_events['meteor_finder'], width, height), daemon=True, name="MeteorCapture"),
                        threading.Thread(target=processing_thread, args=(state_events['meteor_finder'], effective_framerate, pre_event_buffer, width, height), daemon=True, name="MeteorProcess"),
                        threading.Thread(target=writer_thread, args=(state_events['meteor_finder'],), daemon=True, name="MeteorWriter"),
                        threading.Thread(target=snapshot_writer_thread, args=(state_events['meteor_finder'],), daemon=True, name="MeteorSnapshot")                    
                    ]
                    state_events['meteor_finder'].set()
                    for t in active_threads: t.start()
                
                elif current_state == "TIMELAPSE":
                    logging.info("[MANAGER] Riconfigurazione della camera per TIMELAPSE...")
                    capture_format = "RGB888" if args.timelapse_color else "YUV420"
                    exposure_time_us = int(args.timelapse_exposure * 1000000)
                    timelapse_controls = {
                        "AeEnable": False, "AwbEnable": False,
                        "AnalogueGain": args.timelapse_gain, "ExposureTime": exposure_time_us,
                    }
                    if HAS_AUTOFOCUS:
                        timelapse_controls["AfMode"] = controls.AfModeEnum.Manual
                        timelapse_controls["LensPosition"] = 0.0
                    
                    still_config = picam2.create_still_configuration(main={"size": (width, height), "format": capture_format}, controls=timelapse_controls)
                    picam2.configure(still_config)
                    picam2.start()
                    active_threads = [
                        threading.Thread(target=timelapse_capture_thread, args=(state_events['timelapse'], width, height), daemon=True, name="TimelapseCapture"),
                        threading.Thread(target=timelapse_writer_thread, args=(state_events['timelapse'],), daemon=True, name="TimelapseWriter")
                    ]
                    state_events['timelapse'].set()
                    for t in active_threads: t.start()
                
                elif current_state == "POST_PROCESSING":
                    logging.info("[MANAGER] Ingresso in modalità POST_PROCESSING per la creazione del video timelapse.")
                    timelapse_dir = os.path.join(args.output_dir, "timelapse")
                    video_filename = os.path.join(args.output_dir, f"timelapse_{datetime.now().strftime('%Y%m%d')}.mp4")
                    
                    # Esegui la funzione di creazione video
                    create_timelapse_video(
                        image_folder=timelapse_dir,
                        output_filename=video_filename,
                        fps=args.timelapse_video_fps,
                        cleanup=args.timelapse_cleanup_images
                    )                
                    # Dopo il post-processing, lo stato naturale successivo è IDLE
                    current_state = "IDLE" 
                    logging.info("[MANAGER] Post-processing completato. Ritorno in modalità IDLE.")               
                
                elif current_state == "IDLE":
                    logging.info("[MANAGER] Ingresso in modalità IDLE.")

            # 3. --- Controllo Spegnimento Programmato ---
            if shutdown_time_obj and not shutdown_initiated:
                if datetime.now().time() >= shutdown_time_obj:
                    logging.info(f"[MANAGER] Orario di spegnimento ({args.shutdown_time}) raggiunto. Avvio della terminazione.", extra={'state': 'SHUTDOWN'})
                    shutdown_initiated = True  # Assicura che questo blocco venga eseguito una sola volta
                    shutdown_for_system = True # Flag per eseguire il comando di spegnimento dopo la pulizia
                    running = False            # Avvia la terminazione pulita dello script             
            
            time.sleep(30) # Intervallo di controllo dello scheduler

    except KeyboardInterrupt:
        logging.info("[MAIN] Terminazione richiesta dall'utente...")
    finally:
        running = False
        for event in state_events.values(): event.clear()
        
        # Ensure any final video is saved
        if 'recording_event' in globals() and recording_event.is_set():
            logging.info("[MAIN] Finalizzazione della registrazione in corso...")
            output_queue.put(None)

        all_threads = [monitor_t] + active_threads
        for t in all_threads: t.join(timeout=5)
        
        if picam2.started: picam2.stop()
        
        # This final check is belt-and-suspenders, but safe
        if out: out.release()
        if ffmpeg_proc:
            if ffmpeg_proc.stdin:
                try: ffmpeg_proc.stdin.close()
                except BrokenPipeError: pass
            ffmpeg_proc.wait()
        
        logging.info("[MAIN] Uscita completata.")

    # --- Fase di Spegnimento del Sistema ---
    # Questo blocco viene eseguito solo dopo che il blocco `finally` è completato
    if shutdown_for_system:
        logging.info("[SHUTDOWN] Esecuzione del comando di spegnimento del sistema (sudo shutdown now)...", extra={'state': 'SHUTDOWN'})
        logging.info("**************************************************", extra={'state': 'SHUTDOWN'})
        logging.info(f"[SHUTDOWN] Spegnimento del sistema programmato per le {args.shutdown_time} in corso...", extra={'state': 'SHUTDOWN'})
        logging.info("**************************************************", extra={'state': 'SHUTDOWN'})
        # Svuota i buffer del sistema operativo per assicurarsi che i log siano scritti su disco
        os.sync() 
        time.sleep(2) # Breve attesa per sicurezza
        
        # Esegui il comando di spegnimento
        # NOTA: Lo script deve essere eseguito con `sudo` affinché questo comando funzioni.
        os.system("sudo shutdown now")
