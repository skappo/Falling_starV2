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

# === CONTROLLO DIPENDENZE ===
REQUIRED_PACKAGES = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "picamera2": "picamera2",
    "ffmpeg": None  # controllato esternamente
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
            json.dump(vars(args), f, indent=4)
        print(f"[CONFIG] Configurazione aggiornata e salvata in {CONFIG_FILE}")
    except Exception as e:
        print(f"[ERROR] Could not save config to {CONFIG_FILE}: {e}")
         
# === ARGOMENTI CLI E CONFIGURAZIONE CENTRALE ===
parser = argparse.ArgumentParser(description="Rilevamento eventi luminosi e timelapse con Picamera2", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# --- Master defaults ---
master_defaults = {
    "size": "medium", "binning": True, "gain": 8.0, "codec": "h264", "output_dir": "output",
    "framerate": 30 if pi_model == "pi5" else 20,
    # Meteor Finder settings
    "meteor_start_time": "22:00", "meteor_end_time": "05:00",
    "strategy": "contour", "record_duration": 10, "pre_event_seconds": 5,
    "min_brightness": 180, "min_area": 15,
    "diff_threshold": 25, "trigger_area": 4000, "learning_rate": 0.01,
    # Timelapse settings
    "timelapse_start_time": None, "timelapse_end_time": None,
    "timelapse_exposure": 15, "timelapse_interval": 20,
    "timelapse_gain": 8.0, "timelapse_color": False,
    # Performance & Logging
    "no_log_events": False,
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
parser.add_argument("--output-dir", type=str, help="Directory principale per salvare tutti i file.")
parser.add_argument("--framerate", type=int, help="Framerate target per il rilevamento meteore.")
# Meteor Finder
parser.add_argument("--meteor-start-time", type=str, help="Orario di inizio per il rilevamento meteore (HH:MM).")
parser.add_argument("--meteor-end-time", type=str, help="Orario di fine per il rilevamento meteore (HH:MM).")
parser.add_argument("--strategy", choices=["contour", "diff"], help="Strategia di rilevamento meteore.")
parser.add_argument("--record-duration", type=int, help="Durata della registrazione di un evento meteora.")
parser.add_argument("--pre-event-seconds", type=int, help="Secondi da bufferizzare prima di un evento meteora.")
parser.add_argument("--min-brightness", type=int, help="[Contour] Soglia di luminosità.")
parser.add_argument("--min-area", type=int, help="[Contour] Area minima del contorno.")
parser.add_argument("--diff-threshold", type=int, help="[Diff] Soglia di differenza tra pixel.")
parser.add_argument("--trigger-area", type=int, help="[Diff] Numero di pixel cambiati per attivare.")
parser.add_argument("--learning-rate", type=float, help="[Diff] Tasso di adattamento del reference frame.")
# Timelapse
parser.add_argument("--timelapse-start-time", type=str, help="Orario di inizio per il timelapse (HH:MM).")
parser.add_argument("--timelapse-end-time", type=str, help="Orario di fine per il timelapse (HH:MM).")
parser.add_argument("--timelapse-exposure", type=int, help="[Timelapse] Tempo di esposizione in secondi per ogni scatto.")
parser.add_argument("--timelapse-interval", type=int, help="[Timelapse] Intervallo in secondi tra uno scatto e l'altro.")
parser.add_argument("--timelapse-gain", type=float, help="[Timelapse] Gain analogico (ISO) per gli scatti del timelapse.")
parser.add_argument("--timelapse-color", action="store_true", help="[Timelapse] Salva le immagini del timelapse a colori.")
# Performance & Logging
parser.add_argument("--no-log-events", action="store_true", help="Disabilita il logging.")
parser.add_argument("--frame-queue-maxsize", type=int, help="Dimensione massima della coda di analisi meteore.")
parser.add_argument("--output-queue-maxsize", type=int, help="Dimensione massima della coda di scrittura video meteore.")
parser.add_argument("--snapshot-queue-maxsize", type=int, help="Dimensione massima della coda di scrittura snapshot meteore.")
parser.add_argument("--downscale-factor", type=int, help="Fattore di ridimensionamento per lo stream lores (rilevamento meteore).")

# Carica il file config.json se esiste. I suoi valori sovrascrivono i master_defaults.
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        try:
            config_from_file = json.load(f)
            parser.set_defaults(**config_from_file)
            print(f"[CONFIG] Caricata configurazione da {CONFIG_FILE}")
        except Exception as e:
            print(f"[WARNING] Impossibile analizzare {CONFIG_FILE}: {e}")

# Se ci sono argomenti dalla riga di comando sovrascrivono TUTTO il resto.
args = parser.parse_args()

# Salva la configurazione finale
save_config(args)

# Imposta la strategia di riconoscimento
STRATEGY = args.strategy

# === LOGGING ===
# Configura il logging sulla console e su un file `detection_log.log`.
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO, force=True)
if args.no_log_events:
    logging.disable(logging.CRITICAL)

log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler = logging.FileHandler("detection_log.log")
file_handler.setFormatter(log_formatter)
logging.getLogger().addHandler(file_handler)

# === PARAMETRI DERIVATI ===
RESOLUTIONS = {"small": (640, 480), "medium": (1280, 720), "large": (1920, 1080)}

# === ESPOSIZIONE ===
METEOR_EXPOSURE_LIMITS = (50000, 1000000)

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

# === FUNZIONI DI SCHEDULING ===
def parse_time(t):
    return datetime.strptime(t, "%H:%M").time() if t else None

start_time = parse_time(args.start_time)
stop_time = parse_time(args.stop_time)

def should_run_now():
    now = datetime.now().time()
    if not start_time and not stop_time: return True
    if start_time and stop_time:
        if start_time < stop_time: return start_time <= now < stop_time
        else: return now >= start_time or now < stop_time # Gestisce l'intervallo a cavallo della mezzanotte
    if start_time: return now >= start_time
    if stop_time: return now < stop_time
    return True

# === INIZIALIZZAZIONE CAMERA (Iniziale) ===
os.makedirs(args.output_dir, exist_ok=True)
picam2 = Picamera2()
HAS_AUTOFOCUS = is_autofocus_camera(picam2) # Rileva una volta all'avvio
width, height = RESOLUTIONS[args.size]

# === SUPPORTO H264 PER RASPBERRY PI ===
def start_ffmpeg_writer(filename, width, height, framerate):
    # Avvia un processo ffmpeg che accetta dati raw in grayscale e li codifica in H.264 su un raspberry utilizzando h264_v4l2m2m. 
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "gray",
        "-s", f"{width}x{height}", "-r", str(framerate), "-i", "-", "-an",
        "-vcodec", "h264_v4l2m2m", # <-- USA L'ENCODER HARDWARE del Raspberry Pi
        "-preset", "ultrafast", "-crf", "23", filename
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)

# === CODEC ===
# Determina l'estensione del file e il FourCC per i codec non-H264.
if args.codec == "avi":
    fourcc = cv2.VideoWriter_fourcc(*'IYUV')
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
pre_event_buffer = deque(maxlen=int(args.pre_event_seconds * args.framerate))
reference_frame = None
recording_event = threading.Event()
out, ffmpeg_proc = None, None
running = True
current_state = "IDLE"
last_event_time = 0
record_start_time = 0
frames_captured, frames_processed, frames_written, events_triggered = 0, 0, 0, 0
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
def capture_thread_meteor(state_event):
    """Cattura frame ad alto framerate per il rilevamento meteore."""
    while running and state_event.is_set():
        main_frame, lores_frame = picam2.capture_arrays(["main", "lores"])
        try:
            frame_queue.put_nowait((main_frame, lores_frame))
        except queue.Full:
            pass

def monitor_thread():
    """Stampa aggiornamenti periodici sullo stato."""
    while running:
        q_sizes = f"[{frame_queue.qsize()}/{output_queue.qsize()}/{snapshot_queue.qsize()}/{timelapse_writer_queue.qsize()}]"
        print(f"\n[MONITOR] Stato @ {datetime.now().strftime('%H:%M:%S')}: {current_state}, Code [Meteor/Video/Snap/Timelapse]: {q_sizes}")
        time.sleep(15)

# --- Meteor Finder Threads ---
def processing_thread(state_event):
    # Preleva i frame, esegue il rilevamento e gestisce la registrazione.
    global reference_frame, last_event_time, out, ffmpeg_proc, record_start_time
    # Calcola la soglia di area per lo stream a bassa risoluzione.
    scaled_trigger_area = args.trigger_area / (args.downscale_factor**2)
    while running:
        if not should_run_now():
            time.sleep(10)
            continue
        try:
            # Prende i 2 frame (alta e bassa risoluzione) dalla coda.
            full_res_frame, detection_frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue
        
        # Analizza quello a bassa risoluzione (`detection_frame`).
        blurred = cv2.medianBlur(detection_frame, 3)
        motion_detected = False

        if STRATEGY == "diff":
            # Logica basata sulla differenza tra frame.
            if reference_frame is None:
                reference_frame = blurred.copy().astype("float")
                continue
            frame_diff = cv2.absdiff(blurred, cv2.convertScaleAbs(reference_frame))
            _, thresh = cv2.threshold(frame_diff, args.diff_threshold, 255, cv2.THRESH_BINARY)
            changed_area = cv2.countNonZero(thresh)
            motion_detected = changed_area > scaled_trigger_area
            cv2.accumulateWeighted(blurred, reference_frame, args.learning_rate)
        else:
            # Logica basata sulla ricerca di oggetti luminosi.
            _, thresh = cv2.threshold(blurred, args.min_brightness, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_detected = any(cv2.contourArea(c) > args.min_area for c in contours)

        # Aggiunge il frame ad alta risoluzione al buffer circolare del pre-evento.
        pre_event_buffer.append(full_res_frame.copy())
        current_time = time.time()

        # Se viene rilevato un evento e non stiamo già registrando...
        if motion_detected and not recording_event.is_set() and (current_time - last_event_time > 2):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(args.output_dir, f"evento_{timestamp}.{extension}")
            snapshot = os.path.join(args.output_dir, f"snapshot_{timestamp}.jpg")
            
            # Mette una richiesta di salvataggio snapshot nella coda.
            try:
                snapshot_queue.put((snapshot, full_res_frame.copy()), block=False)
            except queue.Full:
                logging.warning("[PROCESS] Coda snapshot piena, snapshot scartato.")            
            
            # Avvia la registrazione video (ffmpeg o OpenCV).
            if args.codec == "h264":
                ffmpeg_proc = start_ffmpeg_writer(filename, full_res_frame.shape[1], full_res_frame.shape[0], FRAME_RATE)
            else:
                out = cv2.VideoWriter(filename, fourcc, FRAME_RATE, (full_res_frame.shape[1], full_res_frame.shape[0]), isColor=False)
            
            # Sposta i frame dal buffer del pre-evento nella coda di output per la scrittura.
            for f in pre_event_buffer:
                try:
                    output_queue.put(f, timeout=0.1)
                except queue.Full:
                    logging.warning("[PROCESS] Coda di output piena durante la scrittura del pre-evento.")

            # Imposta lo stato di registrazione.
            recording_event.set()
            record_start_time = current_time
            last_event_time = current_time
            update_perf_counter("events")
            logging.info(f"[EVENT] Evento rilevato: registrazione iniziata {filename}")

        # Se siamo in stato di registrazione...
        elif recording_event.is_set():
            # Aggiunge il frame corrente alla coda di output.
            try:
                output_queue.put(full_res_frame, timeout=1)
            except queue.Full:
                logging.warning("[PROCESS] Coda di output piena, frame perso")

            # Controlla se la durata della registrazione è terminata.
            if time.time() - record_start_time > args.record_duration:
                # Chiude i file e resetta lo stato.
                if ffmpeg_proc:
                    if ffmpeg_proc.stdin:
                        try: ffmpeg_proc.stdin.close()
                        except BrokenPipeError: pass
                    ffmpeg_proc.wait()
                    ffmpeg_proc = None
                if out:
                    out.release()
                    out = None
                recording_event.clear()
                logging.info("[EVENT] Registrazione terminata")

def writer_thread(state_event):
    # Gestisce la scrittura dei frame video su disco.
    global out, ffmpeg_proc
    while running:
        try:
            frame = output_queue.get(timeout=1)
        except queue.Empty:
            continue
        if recording_event.is_set():
            try:
                if ffmpeg_proc:
                    ffmpeg_proc.stdin.write(frame.tobytes())
                elif out:
                    out.write(frame)
                update_perf_counter("written")
            except BrokenPipeError:
                logging.error("[WRITE] ffmpeg chiuso inaspettatamente.")
                recording_event.clear()
            except Exception as e:
                logging.error(f"[WRITE] Errore scrittura frame: {e}")

def snapshot_writer_thread(state_event):
    # Gestisce la scrittura degli snapshot su disco.
    while running:
        try:
            snapshot_path, frame_to_save = snapshot_queue.get(timeout=1)
            start = time.time()
            cv2.imwrite(snapshot_path, frame_to_save)
            duration = time.time() - start
            logging.info(f"[SNAPSHOT] Snapshot salvato in {snapshot_path} ({duration:.3f}s)")
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"[SNAPSHOT] Errore nel salvataggio snapshot: {e}")

# --- Timelapse Threads ---
def timelapse_capture_thread(state_event):
    """Cattura scatti a lunga esposizione a intervalli regolari."""
    logging.info("[TIMELAPSE] Thread di cattura avviato.")
    last_capture_time = time.time() - args.timelapse_interval # Per scattare subito la prima foto

    while running and state_event.is_set():
        if time.time() - last_capture_time >= args.timelapse_interval:
            last_capture_time = time.time()
            try:
                logging.info("[TIMELAPSE] Cattura di un'immagine a lunga esposizione...")
                captured_frame = picam2.capture_array()
                
                if args.timelapse_color:
                    final_frame = captured_frame
                else:
                    final_frame = captured_frame[:height, :width]
                
                timelapse_writer_queue.put(final_frame)
                logging.info("[TIMELAPSE] Immagine catturata e inviata per il salvataggio.")
            except Exception as e:
                logging.error(f"[TIMELAPSE] Errore durante la cattura: {e}")
        time.sleep(1) # Breve pausa per ridurre il carico della CPU
    logging.info("[TIMELAPSE] Thread di cattura terminato.")

def timelapse_writer_thread(state_event):
    """Salva su disco le immagini del timelapse."""
    timelapse_dir = os.path.join(args.output_dir, "timelapse")
    os.makedirs(timelapse_dir, exist_ok=True)
    logging.info(f"[TIMELAPSE] Thread di scrittura avviato. Salvataggio in: {timelapse_dir}")
    
    while running and state_event.is_set():
        try:
            frame = timelapse_writer_queue.get(timeout=1)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(timelapse_dir, f"tl_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
        except queue.Empty:
            continue
    logging.info("[TIMELAPSE] Thread di scrittura terminato.")

# === MAIN: The State Machine Manager ===
if __name__ == "__main__":
    active_threads = []
    state_events = {'meteor_finder': threading.Event(), 'timelapse': threading.Event()}

    # Avvia i thread universali
    monitor_t = threading.Thread(target=monitor_thread, daemon=True)
    monitor_t.start()
    
    try:
        while running:
            # 1. Determina lo stato desiderato in base alla pianificazione
            desired_state = "IDLE"
            if is_time_in_interval(args.meteor_start_time, args.meteor_end_time):
                desired_state = "METEOR_FINDER"
            elif is_time_in_interval(args.timelapse_start_time, args.timelapse_end_time):
                desired_state = "TIMELAPSE"

            # 2. Se lo stato deve cambiare, esegui la transizione
            if desired_state != current_state:
                logging.info(f"[MANAGER] Transizione di stato: {current_state} -> {desired_state}")

                # --- Fase di SHUTDOWN ---
                if current_state != "IDLE":
                    logging.info(f"[MANAGER] Arresto della modalità {current_state}...")
                    state_events[current_state.lower()].clear()
                    for t in active_threads: t.join(timeout=5)
                    active_threads.clear()
                    picam2.stop()
                    logging.info(f"[MANAGER] Modalità {current_state} arrestata.")
                
                # --- Fase di STARTUP ---
                current_state = desired_state
                if current_state == "METEOR_FINDER":
                    logging.info("[MANAGER] Riconfigurazione della camera per METEOR_FINDER...")
                    lores_w = width // args.downscale_factor
                    lores_h = height // args.downscale_factor
                    video_config = picam2.create_video_configuration(
                        main={"size": (width, height), "format": "Y"},
                        lores={"size": (lores_w, lores_h), "format": "Y"},
                        controls={"FrameDurationLimits": METEOR_EXPOSURE_LIMITS, "AnalogueGain": args.gain, "FrameRate": FRAME_RATE}
                    )
                    picam2.configure(video_config)
                    picam2.start()
                    if args.framerate_mode == 'dynamic':
                        # In modalità dinamica, interroga i metadati della camera per scoprire il framerate reale
                        metadata = picam2.capture_metadata()
                        FRAME_RATE = metadata["FrameRate"]
                        logging.info(f"[CAMERA] Framerate dinamico rilevato: {FRAME_RATE:.2f} fps")
                    else:
                        # In modalità fissa, usa il valore specificato dall'utente
                        FRAME_RATE = args.framerate
                        logging.info(f"[CAMERA] Framerate fisso impostato a: {FRAME_RATE} fps")
                        
                    active_threads = [
                        threading.Thread(target=capture_thread_meteor, args=(state_events['meteor_finder'],), daemon=True),
                        threading.Thread(target=processing_thread, args=(state_events['meteor_finder'],), daemon=True),
                        threading.Thread(target=writer_thread, args=(state_events['meteor_finder'],), daemon=True),
                        threading.Thread(target=snapshot_writer_thread, args=(state_events['meteor_finder'],), daemon=True)
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
                        threading.Thread(target=timelapse_capture_thread, args=(state_events['timelapse'],), daemon=True),
                        threading.Thread(target=timelapse_writer_thread, args=(state_events['timelapse'],), daemon=True)
                    ]
                    state_events['timelapse'].set()
                    for t in active_threads: t.start()
                
                elif current_state == "IDLE":
                    logging.info("[MANAGER] Ingresso in modalità IDLE.")

            time.sleep(30) # Intervallo di controllo dello scheduler

    except KeyboardInterrupt:
        logging.info("[MAIN] Terminazione richiesta dall'utente...")
    finally:
        running = False
        for event in state_events.values(): event.clear()
        all_threads = [monitor_t] + active_threads
        for t in all_threads: t.join(timeout=2)
        if picam2.started: picam2.stop()
        logging.info("[MAIN] Uscita completata.")
