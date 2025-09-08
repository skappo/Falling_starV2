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
import os
from datetime import datetime
from collections import deque
from picamera2 import Picamera2
#from picamera2.encoders import H264Encoder
from libcamera import controls
from colorama import Fore, Style, Back, init

# --- Global Configuration Variable ---
# This dictionary will replace the `args` object and hold all runtime settings.
APP_CONFIG = {}

# === CONTROLLO DIPENDENZE ===
# Blocco di controllo per garantire che tutte le librerie necessarie siano installate
# prima di avviare lo script. Fornisce messaggi chiari e comandi di installazione.
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

# === PARAMETRI DERIVATI (Definitive Final Version) ===
# Queste costanti sono definite globalmente in modo che siano accessibili
# da tutte le funzioni, incluso l'editor di configurazione.
RESOLUTIONS = {"small": (640, 480), "medium": (1280, 720), "large": (1920, 1080)}
METEOR_EXPOSURE_LIMITS = (50000, 1000000)
CHOICES = {
    "size": ["small", "medium", "large"],
    "codec": ["avi", "mjpeg", "h264"],
    "strategy": ["contour", "diff"],
    "framerate_mode": ["fixed", "dynamic"]
}

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
def save_config(config_dict):
    """Salva un dizionario di configurazione in formato JSON."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config_dict, f, indent=4)
        print(f"{Fore.GREEN}[CONFIG] Configurazione salvata con successo in {CONFIG_FILE}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}[ERROR] Impossibile salvare la configurazione in {CONFIG_FILE}: {e}{Style.RESET_ALL}")

# === EDITOR INTERATTIVO DELLA CONFIGURAZIONE (Now the main entry point) ===
def resolve_resolution(cfg):
    """Restituisce (w, h) in base a cfg['size'] o cfg['resolution'].
    Precedenza a 'size' (stringa: small/medium/large). In alternativa, usa
    'resolution' che può essere una stringa (chiave di RESOLUTIONS) oppure una
    tupla/lista (w, h). Fallback: 'medium'."""
    size_key = cfg.get('size')
    if isinstance(size_key, str) and size_key in RESOLUTIONS:
        return RESOLUTIONS[size_key]

    res = cfg.get('resolution', 'medium')
    if isinstance(res, str) and res in RESOLUTIONS:
        return RESOLUTIONS[res]
    if isinstance(res, (tuple, list)) and len(res) == 2:
        return tuple(res)

    return RESOLUTIONS['medium']  # fallback


def update_autocalculated(cfg):
    """
    Aggiorna i valori auto-calcolati in base alla configurazione corrente.
    - Calcola frame_size_bytes assumendo immagini in scala di grigi (Y from YUV): 1 byte/pixel
    - Ricalcola frame_queue_maxsize, output_queue_maxsize e estimated_memory_mb
    """
    megabyte = 1048576
    safety_cushion = 1.05
    try:
        framerate = cfg.get('framerate', 0) or 0
        pre_event = cfg.get('pre_event_seconds', 0) or 0
        record_duration = cfg.get('record_duration', 0) or 0

        # Calcolo dimensione frame in base a size o resolution
        w, h = resolve_resolution(cfg)
        frame_size_bytes = w * h # Grayscale (1 byte per pixel)
        cfg['frame_size_bytes'] = frame_size_bytes

        # Code
        cfg['frame_queue_maxsize'] = int(framerate * 1.5 * safety_cushion)
        cfg['output_queue_maxsize'] = int((pre_event + record_duration) * framerate * safety_cushion)
        
        # Uso memoria stimato
        cfg['frame_queue_mb'] = round((cfg['frame_queue_maxsize'] * frame_size_bytes) / megabyte, 2)
        cfg['output_queue_mb'] = round((cfg['output_queue_maxsize'] * frame_size_bytes) / megabyte, 2)

        # Uso memoria stimato
 #       total_frames = cfg['frame_queue_maxsize'] + cfg['output_queue_maxsize']
 #       cfg['estimated_memory_mb'] = round((total_frames * frame_size_bytes) / (1024 * 1024), 2)

    except Exception:
#        cfg['frame_size_bytes'] = 0
        cfg['frame_queue_maxsize'] = 0
        cfg['output_queue_maxsize'] = 0
#        cfg['estimated_memory_mb'] = 0.0
        cfg['frame_queue_mb'] = 0.0
        cfg['output_queue_mb'] = 0.0


def edit_config_interactive(current_config):
    """
    Mostra un menu interattivo per modificare la configurazione, con validazione,
    calcolo dinamico delle code e visualizzazione delle scelte.
    """
    cfg = current_config.copy()
    update_autocalculated(cfg)

    while True:
        clear_screen()
        print("\n" + "="*15 + " MENU CONFIGURAZIONE " + "="*15)

        # Stampa i parametri in modo leggibile
        keys = list(cfg.keys())
        for i, key in enumerate(keys, start=1):
            value = cfg[key]

            value_str = f"{Fore.YELLOW}{value}{Style.RESET_ALL}"
            if value is None:
                value_str = f"{Fore.LIGHTBLACK_EX}Non impostato{Style.RESET_ALL}"
            elif isinstance(value, bool):
                value_str = f"{Fore.CYAN}{value}{Style.RESET_ALL}"
            else:
                value_str = f"{Fore.LIGHTGREEN_EX}{value}{Style.RESET_ALL}"                 

            # Parametri auto-calcolati personalizzati
            if key == 'frame_queue_maxsize':
                extra = f"{cfg.get('frame_queue_mb')})"
                print(f"  {Fore.LIGHTBLACK_EX}{i:2d}. {key:<28}{Style.RESET_ALL} = {value_str} (Auto-Calcolato: {extra}")
                continue
            elif key == 'output_queue_maxsize':
                extra = f"{cfg.get('output_queue_mb')} MB)"
                print(f"  {Fore.LIGHTBLACK_EX}{i:2d}. {key:<28}{Style.RESET_ALL} = {value_str} (Auto-Calcolato: {extra}")
                continue
            elif key in ['estimated_memory_mb', 'frame_size_bytes', 'resolution', 'frame_queue_mb', 'output_queue_mb' ]:
                # non visualizzare separatamente
                continue

            # Scelte predefinite
            choices_str = ""
            if key in CHOICES:
                choices_str = f" (Scelte: {Fore.GREEN}{', '.join(CHOICES[key])}{Style.RESET_ALL})"
            elif key == "timelapse_exposure":
                choices_str = f" (Scelte: {Fore.GREEN}<numero>, 'automatic'{Style.RESET_ALL})"

            print(f"  {Style.BRIGHT}{i:2d}. {key:<28}{Style.RESET_ALL} = {value_str}{choices_str}")

#        print(f"\n  {Style.BRIGHT}{Fore.GREEN} 0. Avvia Programma{Style.RESET_ALL}")
#        choice = input(f"\nScegli un numero da modificare, {Fore.GREEN}0 per avviare{Style.RESET_ALL}, o lascia vuoto per uscire: ")

        # Aggiunge le opzioni di avvio e test
        print(f"\n  {Style.BRIGHT}{Fore.GREEN} 0. Avvia Programma{Style.RESET_ALL}")
        print(f"  {Style.BRIGHT}{Fore.YELLOW} V. Gira Video di Prova (5s){Style.RESET_ALL}")
        print(f"  {Style.BRIGHT}{Fore.YELLOW} P. Scatta Foto di Prova (Timelapse){Style.RESET_ALL}")
        print(f"  {Style.BRIGHT}{Fore.MAGENTA} A. Esegui Test Automatico Camera{Style.RESET_ALL}")      
        
        choice = input(f"\nScegli un numero da modificare, o lascia vuoto per uscire: ")

        if not choice.strip():
#            confirm = input("Sei sicuro di voler uscire senza salvare? (s/N): ").lower()
#            if confirm == 's':
                return None
#            else:
#                continue

        if choice == "0":
            update_autocalculated(cfg)
            print("[INFO] Salvataggio della configurazione finale...")
            save_config(cfg)
            return cfg
        elif choice.lower() == 'v':
            run_video_test(cfg)
            continue
        elif choice.lower() == 'p':
            run_photo_test(cfg)
            continue
        elif choice.lower() == 'a':
            run_auto_test()
            # Dopo il test, il loop continuerà, mostrando di nuovo il menu.
            continue
            
        if choice.isdigit() and 1 <= int(choice) <= len(keys):
            key_to_edit = keys[int(choice) - 1]

            # Blocco modifica per parametri auto-calcolati
            if key_to_edit in ['frame_queue_maxsize', 'output_queue_maxsize', 'estimated_memory_mb', 'frame_size_bytes', 'resolution']:
                print(f"\n{Fore.YELLOW}[INFO] Il parametro '{key_to_edit}' è calcolato automaticamente.{Style.RESET_ALL}")
                time.sleep(2)
                continue

            current_value = cfg[key_to_edit]
            new_val_str = input(f"Nuovo valore per '{key_to_edit}' (attuale: {current_value}): ")

            if not new_val_str.strip():
                print("[INFO] Nessuna modifica effettuata.")
                time.sleep(1)
                continue

            # Validazione speciale per timelapse_exposure
            if key_to_edit == "timelapse_exposure":
                if new_val_str.lower() == 'automatic':
                    cfg[key_to_edit] = 'automatic'
                elif new_val_str.isdigit():
                    cfg[key_to_edit] = int(new_val_str)
                else:
                    print(f"{Fore.RED}[ERRORE] Valore non valido. Inserire un numero o 'automatic'.{Style.RESET_ALL}")
                    time.sleep(2)
                    continue
                print(f"{Fore.GREEN}[OK] Valore aggiornato: {key_to_edit} = {cfg[key_to_edit]}{Style.RESET_ALL}")
                update_autocalculated(cfg)
                time.sleep(1)
                continue

            # Validazione per parametri con scelte
            if key_to_edit in CHOICES and new_val_str not in CHOICES[key_to_edit]:
                print(f"{Fore.RED}[ERRORE] Valore non valido. Le scelte possibili sono: {', '.join(CHOICES[key_to_edit])}{Style.RESET_ALL}")
                time.sleep(2)
                continue

            # Conversione generica
            try:
                if isinstance(current_value, bool):
                    new_val = new_val_str.lower() in ['true', '1', 't', 'y', 's']
                elif new_val_str.lower() in ['none', 'null']:
                    new_val = None
                elif isinstance(current_value, int):
                    new_val = int(float(new_val_str))
                elif isinstance(current_value, float):
                    new_val = float(new_val_str)
                else:
                    new_val = new_val_str

                cfg[key_to_edit] = new_val
                update_autocalculated(cfg)
                print(f"{Fore.GREEN}[OK] Valore aggiornato: {key_to_edit} = {new_val}{Style.RESET_ALL}")

            except (ValueError, TypeError):
                print(f"{Fore.RED}[ERRORE] Input non valido per il tipo atteso.{Style.RESET_ALL}")
                time.sleep(2)
        else:
            print(f"{Fore.RED}[ERRORE] Scelta non valida.{Style.RESET_ALL}")
            time.sleep(1)
        
# === LOGGING CLASS ===
class ColoredFormatter(logging.Formatter):
    """
    Formatter personalizzato che applica colori specifici per data, livello e tag,
    con una palette ottimizzata per sfondi scuri.
    """
    DATETIME_COLOR = Fore.LIGHTBLACK_EX
    MESSAGE_COLOR = Fore.WHITE
    RESET = Style.RESET_ALL

    LEVEL_COLORS = {
        "INFO": Fore.LIGHTWHITE_EX,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED + Style.BRIGHT,
        "DEBUG": Fore.LIGHTBLACK_EX,
    }

    TAG_COLORS = {
        "[MAIN]": Fore.CYAN + Style.BRIGHT,
        "[MONITOR]": Fore.CYAN,
        "[MANAGER]": Fore.LIGHTBLUE_EX + Style.BRIGHT,
        "[CAMERA]": Fore.LIGHTBLUE_EX,
        "[PROCESS]": Fore.LIGHTCYAN_EX,
        "[WRITE]": Fore.YELLOW,
        "[EVENT]": Fore.GREEN,
        "[SNAPSHOT]": Fore.GREEN,
        "[TIMELAPSE]": Fore.LIGHTCYAN_EX,
        "[POST-PROCESSING]": Fore.LIGHTYELLOW_EX,
        "[SHUTDOWN]": Fore.RED + Style.BRIGHT,
    }

    STATE_COLORS = {
        "METEOR_FINDER": Fore.LIGHTBLUE_EX,
        "TIMELAPSE": Fore.CYAN,
        "POST_PROCESSING": Fore.YELLOW,
        "IDLE": Fore.LIGHTBLACK_EX
    }

    def format(self, record):
        # Crea il messaggio di log di base
        log_message = super().format(record)
        
        # Gestisce il colore speciale per lo stato nel messaggio del monitor
        if "[MONITOR]" in log_message and hasattr(record, 'state'):
            state_str = record.state
            color = self.STATE_COLORS.get(state_str, Fore.WHITE)
            log_message = log_message.replace(state_str, f"{color}{state_str}{self.RESET}")

        # Applica i colori per i tag
        for tag, color in self.TAG_COLORS.items():
            if tag in log_message:
                log_message = log_message.replace(tag, f"{color}{tag}{self.RESET}")
        
        # Suddivide il messaggio per colorare data e livello in modo indipendente
        parts = log_message.split(" ", 2)
        if len(parts) >= 3:
            datetime_str = f"{parts[0]} {parts[1]}"
            rest = parts[2]
            
            # Colora il livello del log
            level_str = rest.split("]")[0] + "]"
            level_color = self.LEVEL_COLORS.get(record.levelname, "")
            rest = rest.replace(level_str, f"{level_color}{level_str}{self.RESET}")
            
            return f"{self.DATETIME_COLOR}{datetime_str}{self.RESET} {rest}"
        
        return log_message # Fallback

# === CREAZIONE VIDEO TIMELAPSE ===
def create_timelapse_video(image_folder, output_filename, fps, cleanup=False):
    # Utilizza ffmpeg per creare un video da una sequenza di immagini.
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
    # Controlla se l'ora corrente rientra in un dato intervallo (gestisce la mezzanotte).
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
    # Camera Info, if v3 disable autofocus
    try:
        model = picam2_object.camera_properties.get('Model', 'unknown').lower()
        if 'imx708' in model:
            logging.info(f"Rilevato sensore IMX708 (Camera Module 3). Verranno applicati i controlli di fuoco manuale per il timelapse.")
            return True
    except Exception as e:
        logging.error(f"Impossibile determinare il modello della camera: {e}")
    return False

# === SUPPORTO H264 PER RASPBERRY PI ===
def start_ffmpeg_writer(filename, width, height, framerate):
    """
    Costruisce e avvia il comando ffmpeg per la registrazione video.
    - Accetta input GRAYSCALE per mantenere semplice la pipeline Python.
    - Usa un filtro video (`-vf`) per convertire in YUV420p, il formato richiesto
      dall'encoder hardware `h264_v4l2m2m`.
    """
    cmd = [
        "nice", "-n", "10",
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
    THREAD CRITICO: Produttore per la modalità METEOR_FINDER.
    - Cattura due stream (alta e bassa risoluzione) dalla camera.
    - Usa il pattern `capture_request` per garantire dati validi e stabili.
    - Converte immediatamente entrambi gli stream in grayscale per efficienza.
    - Usa `put_nowait` per scartare frame se la pipeline è piena, evitando blocchi.
    """
    # Pre-calcola le dimensioni dello stream a bassa risoluzione per lo slicing
    lores_w = width // APP_CONFIG['downscale_factor']
    lores_h = height // APP_CONFIG['downscale_factor']
    
    while running and state_event.is_set():
        job = None
        request = None # Inizializza a None per sicurezza
        try:
            # 1. Avvia la cattura e ottieni un "job". Questo non è bloccante.
            job = picam2.capture_request(wait=False)

            # 2. Attendi il completamento del job.
            #    picam2.wait() restituisce un oggetto CompletedRequest in caso di successo,
            #    o None in caso di timeout.
            request = picam2.wait(job, timeout=1000) # Timeout di 1 secondo

#            if request is None:
#                # Se la richiesta scade, logga un avviso e continua.
#                # Questo può accadere se il sistema è sotto carico pesante.
#                logging.warning("[CAPTURE] La richiesta di cattura è scaduta (timeout).")
#                continue # Salta al prossimo ciclo del loo

            # 3. Prosegui SOLO se abbiamo ricevuto una richiesta completata con successo.
            if request:
                main_frame_yuv = request.make_array("main")
                lores_frame_yuv = request.make_array("lores")
          
                main_frame_gray = main_frame_yuv[:height, :width]
                lores_frame_gray = lores_frame_yuv[:lores_h, :lores_w]

                frame_queue.put_nowait((main_frame_gray, lores_frame_gray))
                update_perf_counter("captured")
            else:
                # Se la richiesta è None, il timeout è scaduto. Questo è normale durante
                # lo shutdown della camera, quindi non logghiamo un errore.
                logging.warning("[CAPTURE] La richiesta di cattura del frame è scaduta (timeout).")
                continue

        except queue.Full:
            pass # Comportamento previsto
        except Exception as e:
            if "Camera frontend has timed out" not in str(e):
                logging.error(f"[CAPTURE] Errore imprevisto: {e}")
        finally:
            # 4. Rilascia la richiesta SOLO se è un oggetto CompletedRequest valido.
            #    NON tentare mai di rilasciare il "job".
            if request:
                request.release()

def monitor_thread():
    """
    THREAD CRITICO: Stampa lo stato del sistema e i contatori.
    - Funziona in background per fornire un "heartbeat" e metriche di performance.
    - In modalità IDLE, stampa un messaggio semplificato.
    - Calcola i totali di sessione (persistenti) e i tassi attuali (FPS).
    """
    global frames_captured, frames_processed, frames_written, events_triggered, lock_perf, state_lock, current_state
    INTERVAL = 15 # secondi tra una visualizzazione e l'altra

    # Variabili per calcolare i tassi (FPS) tra un intervallo e l'altro
    last_fc, last_fp, last_fw, last_ev = 0, 0, 0, 0

    while running:
        time.sleep(INTERVAL)

        with state_lock:
            local_state = current_state

        # Se lo script è inattivo, non stampare nulla e attendi il prossimo ciclo.
        if local_state == "IDLE":
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
        q_frame = color_queue(frame_queue.qsize(), APP_CONFIG['frame_queue_maxsize'])
        q_out = color_queue(output_queue.qsize(), APP_CONFIG['output_queue_maxsize'])
        q_snap = color_queue(snapshot_queue.qsize(), APP_CONFIG['snapshot_queue_maxsize'])
        q_tl = color_queue(timelapse_writer_queue.qsize(), 10)

        # --- Output (same structure as before, but with colors) ---
        logging.info(f"[MONITOR] Stato @ {datetime.now().strftime('%H:%M:%S')}: {color_state(local_state)}")
        # Stampa i totali incrementali
        logging.info(f"  Totali Sessione -> Frame Acquisiti: {current_fc} | Frame Processati: {current_fp} | Frame Scritti: {current_fw}")
        logging.info(f"  Totali Sessione -> Eventi/Scatti Rilevati: {current_ev}")
        # Stampa i tassi attuali
        logging.info(f"  Tasso Attuale -> FPS (Acq/Proc/Scrit): {fps_cap:.1f}/{fps_proc:.1f}/{fps_write:.1f}")
        logging.info(f"  Stato -> Registrazione Attiva: {status}, Code [Meteora/Video/Snap/Timelapse]: "f"[{q_frame}/{q_out}/{q_snap}/{q_tl}]")

# --- Meteor Finder Threads ---
def processing_thread(state_event, effective_framerate, pre_event_buffer, width, height):
    """
    THREAD CRITICO: Il "cervello" della modalità METEOR_FINDER.
    - Consumatore della `frame_queue`.
    - Esegue la logica di rilevamento sul frame a bassa risoluzione.
    - Gestisce il `pre_event_buffer` con i frame ad alta risoluzione.
    - Orchestra l'inizio di una registrazione.
    - Gestisce la terminazione di una registrazione entrando in uno stato `is_shutting_down`
      per garantire l'invio affidabile del segnale di stop (`None`) al writer.
    """
    global reference_frame, last_event_time, out, ffmpeg_proc, record_start_time

    scaled_trigger_area = APP_CONFIG['trigger_area'] / (APP_CONFIG['downscale_factor']**2)
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
            _, thresh = cv2.threshold(frame_diff, APP_CONFIG['diff_threshold'], 255, cv2.THRESH_BINARY)
            changed_area = cv2.countNonZero(thresh)
            motion_detected = changed_area > scaled_trigger_area
            cv2.accumulateWeighted(blurred, reference_frame, APP_CONFIG['learning_rate'])
        else: # contour strategy
            _, thresh = cv2.threshold(blurred, APP_CONFIG['min_brightness'], 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_detected = any(cv2.contourArea(c) > APP_CONFIG['min_area'] for c in contours)

        # --- Gestione Buffer e Registrazione ---
        pre_event_buffer.append(full_res_frame.copy())
        current_time = time.time()

        # Caso 1: Rilevato un nuovo evento
        if motion_detected and not recording_event.is_set() and (current_time - last_event_time > 2):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(APP_CONFIG['output_dir'], f"evento_{timestamp}.{extension}")
            snapshot_path = os.path.join(APP_CONFIG['output_dir'], f"snapshot_{timestamp}.jpg")

            try:
                snapshot_queue.put((snapshot_path, full_res_frame.copy()), block=False)
            except queue.Full:
                logging.warning("[PROCESS] Coda snapshot piena, snapshot scartato.")

            if APP_CONFIG['codec'] == "h264":
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
            if time.time() - record_start_time > APP_CONFIG['record_duration']:
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
    """
    THREAD CRITICO: Scrittore dei file video per METEOR_FINDER.
    - Consumatore della `output_queue`.
    - Isola la lenta operazione di I/O su disco.
    - Ascolta un segnale `None` (sentinella) per chiudere e finalizzare correttamente
      il file video, prevenendo la corruzione dei dati.
    """
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

                    # Resetta i writer per prevenire ulteriori tentativi di scrittura
                    ffmpeg_proc = None
                    if out:
                        out.release()
                        out = None
                    # NON cancellare il recording_event. Lascia che sia il processing_thread a gestire lo stato.
        except queue.Empty:
            # È normale che la coda sia vuota, continua semplicemente ad attendere.
            time.sleep(0.01)
            continue

    logging.info("[WRITER] Thread di scrittura terminato.")

def snapshot_writer_thread(state_event):
    """Scrittore dei file snapshot per METEOR_FINDER. Deve controllare `state_event` per terminare correttamente."""
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
def timelapse_capture_thread(state_event):
    """
    Cattura scatti a lunga esposizione. Usa il pattern `capture_request`/`wait`
    con un timeout personalizzato per gestire in modo robusto esposizioni lunghe.
    """
    """Cattura scatti a lunga esposizione in modo robusto e a prova di race condition."""
    logging.info("[TIMELAPSE] Thread di cattura avviato.")
    
    is_manual_exposure = str(APP_CONFIG['timelapse_exposure']).lower() != "automatic"
    manual_exposure_time = 0
    if is_manual_exposure:
        try:
            manual_exposure_time = int(APP_CONFIG['timelapse_exposure'])
        except (ValueError, TypeError):
            logging.error(f"[TIMELAPSE] Valore di esposizione manuale non valido: '{APP_CONFIG['timelapse_exposure']}'. Arresto del thread.")
            return

    while running and state_event.is_set():
        job = None
        request = None
        try:
            # --- Fase 1: Cattura ---
            if is_manual_exposure:
                logging.info(f"[TIMELAPSE] Avvio cattura manuale (esposizione: {manual_exposure_time}s)...")
                job = picam2.capture_request(wait=False)
                wait_timeout = manual_exposure_time + 5
                request = picam2.wait(job, timeout=wait_timeout * 1000)
                if request is None: 
                    raise RuntimeError("La richiesta di cattura manuale è scaduta.")
            else:
                logging.info(f"[TIMELAPSE] Avvio cattura automatica...")
                job = picam2.capture_request(wait=False)
                request = picam2.wait(job, timeout=5000) # Timeout di 5s per l'auto-esposizione
                if request is None:
                    raise RuntimeError("La richiesta di cattura automatica è scaduta.")
            
            # --- Fase 2: Processamento (solo se la cattura ha avuto successo) ---
            captured_image = request.make_array('main')
            if APP_CONFIG['timelapse_color']:
                final_frame = cv2.cvtColor(captured_image, cv2.COLOR_RGB2BGR)
            else:
                final_frame = cv2.cvtColor(captured_image, cv2.COLOR_YUV2GRAY_I420)
            
            timelapse_writer_queue.put(final_frame)
            logging.info("[TIMELAPSE] Immagine catturata e inviata per il salvataggio.")

        except Exception as e:
            logging.error(f"[TIMELAPSE] Errore durante la cattura: {e}")
        finally:
            # Rilascia la richiesta SOLO se è un oggetto CompletedRequest valido.
            if request:
                request.release()

        # --- Fase 3: Attesa (Intervallo) ---
        logging.info(f"[TIMELAPSE] Inizio intervallo di attesa di {APP_CONFIG['timelapse_interval']} secondi...")
        for _ in range(APP_CONFIG['timelapse_interval']):
            if not (running and state_event.is_set()):
                break
            time.sleep(1)

    logging.info("[TIMELAPSE] Thread di cattura terminato.")

def timelapse_writer_thread(state_event):
    """Salva su disco le immagini finali del timelapse."""
    timelapse_dir = os.path.join(APP_CONFIG['output_dir'], "timelapse")
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
    
# === FUNZIONI AUSILIARIE ===

def clear_screen():
    """Pulisce lo schermo del terminale per una visualizzazione pulita."""
    os.system('cls' if os.name == 'nt' else 'clear')

def run_auto_test():
    """
    Esegue un test diagnostico della camera per suggerire impostazioni ottimali.
    - Misura il framerate massimo in modalità video grayscale.
    - Esegue uno scatto in modalità completamente automatica per trovare l'esposizione e il gain ideali.
    """
    clear_screen()
    print("\n" + "="*15 + " TEST AUTOMATICO CAMERA " + "="*15)
    print(f"{Fore.YELLOW}Questo test configurerà la camera per misurare le prestazioni.\n"
          f"Assicurati che la camera stia inquadrando la scena desiderata...{Style.RESET_ALL}")
    
    test_picam2 = None
    try:
        # Usa un'istanza locale della camera per non interferire con l'applicazione principale.
        test_picam2 = Picamera2()
        
        # --- Fase 1: Test del Framerate Video ---
        print("\n--- Fase 1: Misurazione Framerate Video ---")
        cfg = {"size": (1920, 1080), "format": "YUV420"}
        video_config = test_picam2.create_video_configuration(main=cfg)
        test_picam2.configure(video_config)
        test_picam2.start()
        logging.info("[TEST] Camera avviata per il test del framerate. Attesa stabilizzazione...")
        time.sleep(1)

        NUM_FRAMES_TO_TEST = 100
        print(f"Cattura di {NUM_FRAMES_TO_TEST} frame il più velocemente possibile...")
        start_time = time.monotonic()
        for _ in range(NUM_FRAMES_TO_TEST):
            test_picam2.capture_array() # Scartiamo i frame, ci interessa solo la velocità
        end_time = time.monotonic()
        
        duration = end_time - start_time
        measured_fps = NUM_FRAMES_TO_TEST / duration
        print(f"Test del framerate completato in {duration:.2f} secondi.")
        test_picam2.stop()

        # --- Fase 2: Test dell'Esposizione Automatica ---
        print("\n--- Fase 2: Rilevamento Esposizione Automatica ---")
        
        # Riconfigura la camera per uno scatto singolo in modalità automatica.
        still_config = test_picam2.create_still_configuration()
        test_picam2.configure(still_config)
        
        # Abilita tutti gli algoritmi automatici della camera.
        test_picam2.set_controls({
            "AeEnable": True,
            "AwbEnable": True
        })
        if is_autofocus_camera(test_picam2):
            test_picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        
        test_picam2.start()
        logging.info("[TEST] Camera avviata per il test dell'esposizione. Attesa convergenza algoritmi...")
        time.sleep(2) # Pausa per dare tempo agli algoritmi di auto-esposizione di stabilizzarsi.

        # Cattura i metadati per vedere quali impostazioni ha scelto la camera.
        metadata = test_picam2.capture_metadata()
        print("Test dell'esposizione completato.")
        
    except Exception as e:
        logging.error(f"[TEST] Si è verificato un errore durante il test: {e}")
    finally:
        if test_picam2 and test_picam2.started:
            test_picam2.stop()
        if test_picam2:
            test_picam2.close()
            logging.info("[TEST] Camera di test chiusa correttamente.")

    # --- Fase 3: Stampa dei Risultati ---
    print("\n" + "="*17 + " RISULTATI DEL TEST " + "="*17)
    if 'measured_fps' in locals():
        print(f"  - Framerate Video Massimo Misurato: {Fore.CYAN}{measured_fps:.1f} FPS{Style.RESET_ALL}")
    if 'metadata' in locals():
        exposure_time = metadata.get("ExposureTime", 0)
        analogue_gain = metadata.get("AnalogueGain", 0)
        print(f"  - Esposizione Automatica Scelta: {Fore.CYAN}{exposure_time / 1000:.1f} ms{Style.RESET_ALL}")
        print(f"  - Gain Analogico (ISO) Scelto: {Fore.CYAN}{analogue_gain:.2f} (equivalente a ISO ~{int(analogue_gain * 100)}){Style.RESET_ALL}")

    print("="*50)
    input("\nPremi Invio per tornare al menu principale...")

def run_video_test(current_config):
    """
    Esegue un test di registrazione video di 5 secondi, creando una pipeline a thread
    completa e accurata (Capture -> Bridge -> Writer) che utilizza l'encoder hardware
    per una simulazione perfetta e affidabile.
    """
    clear_screen()
    print("\n" + "="*15 + " TEST VIDEO (METEOR FINDER) " + "="*15)
    print(f"{Fore.YELLOW}Questo test registrerà un video di 5 secondi con le impostazioni attuali...{Style.RESET_ALL}")
    
    # --- Setup locale e completamente autonomo per il test ---
    test_picam2 = None
    ffmpeg_proc = None
    state_event = threading.Event()
    state_event.set()
    
    # Variabili di stato locali per i thread di test
    test_running = True
    
    try:
        cfg = current_config
        res = RESOLUTIONS[cfg['size']]
        width, height = res[0], res[1]
        
        test_frame_queue = queue.Queue(maxsize=30)
        test_output_queue = queue.Queue(maxsize=120)
        
        test_picam2 = Picamera2()
        
        # Configurazione Video
        meteor_controls = {"FrameDurationLimits": METEOR_EXPOSURE_LIMITS, "AnalogueGain": cfg['gain']}
        framerate = cfg['framerate']
        meteor_controls["FrameRate"] = framerate
            
        video_config = test_picam2.create_video_configuration(
            main={"size": (width, height), "format": "YUV420"},
            lores={"size": (width // cfg['downscale_factor'], height // cfg['downscale_factor'])},
            controls=meteor_controls
        )
        test_picam2.configure(video_config)
        test_picam2.start()
        
        output_dir = cfg['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, "TEST_VIDEO.mp4")
        
        # 1. Avvia il writer ffmpeg con l'encoder hardware.
        ffmpeg_proc = start_ffmpeg_writer(filename, width, height, framerate)
        
        # 2. Definisci i thread di test
        def capture_for_test():
            lores_w = width // cfg['downscale_factor']
            lores_h = height // cfg['downscale_factor']
            while test_running and state_event.is_set():
                job = None
                request = None
                try:
                    job = test_picam2.capture_request(wait=False)
                    request = test_picam2.wait(job, timeout=1000)
                    if request:
                        main_frame_yuv = request.make_array("main")
                        lores_frame_yuv = request.make_array("lores")
                        main_frame_gray = main_frame_yuv[:height, :width]
                        lores_frame_gray = lores_frame_yuv[:lores_h, :lores_w]
                        test_frame_queue.put_nowait((main_frame_gray, lores_frame_gray))
                except queue.Full:
                    pass
                except Exception as e:
                    if test_running and state_event.is_set(): logging.error(f"[TEST_CAPTURE] Errore: {e}")
                finally:
                    if request:
                        request.release()

        def bridge_for_test():
            """Il thread intermedio che disaccoppia la cattura dalla scrittura."""
            while test_running and state_event.is_set():
                try:
                    frame_tuple = test_frame_queue.get(timeout=0.1)
                    test_output_queue.put(frame_tuple[0]) # Passa solo il frame ad alta risoluzione
                except queue.Empty:
                    continue
        
        def writer_for_test():
            while test_running or not test_output_queue.empty():
                try:
                    item = test_output_queue.get(timeout=0.1)
                    if item is None: break
                    if ffmpeg_proc and ffmpeg_proc.stdin:
                        ffmpeg_proc.stdin.write(item.tobytes())
                except queue.Empty:
                    continue
                except Exception as e:
                    logging.error(f"[TEST_WRITER] Errore: {e}")

        # 3. Avvia i thread
        threads = [
            threading.Thread(target=capture_for_test, daemon=True),
            threading.Thread(target=bridge_for_test, daemon=True),
            threading.Thread(target=writer_for_test, daemon=True)
        ]
        
        logging.info(f"[TEST] Avvio registrazione video di test in: {filename}")
        for t in threads: t.start()
        
        time.sleep(5) # Durata del test
        
        logging.info("[TEST] Test completato. Arresto dei thread...")
        
    except Exception as e:
        logging.error(f"[TEST] Si è verificato un errore durante il test video: {e}")
    finally:
        # 4. Esegui una chiusura pulita e garantita.
        test_running = False
        state_event.clear()
        
        if 'test_output_queue' in locals():
            test_output_queue.put(None)

        if 'threads' in locals():
            for t in threads: t.join(timeout=5)
        
        if ffmpeg_proc:
            if ffmpeg_proc.stdin and not ffmpeg_proc.stdin.closed:
                try: ffmpeg_proc.stdin.close()
                except BrokenPipeError: pass
            ffmpeg_proc.wait(timeout=2)
        
        if test_picam2 and test_picam2.started:
            test_picam2.stop()
        if test_picam2:
            test_picam2.close()
            logging.info("[TEST] Camera di test chiusa correttamente.")
    
    input("\nPremi Invio per tornare al menu principale...")

def run_photo_test(current_config):
    """Scatta una singola foto di prova usando le impostazioni TIMELAPSE."""
    clear_screen()
    print("\n" + "="*15 + " TEST FOTO (TIMELAPSE) " + "="*15)
    print(f"{Fore.YELLOW}Questo test scatterà una singola foto con le impostazioni di timelapse attuali...{Style.RESET_ALL}")

    test_picam2 = None
    try:
        cfg = current_config
        res = RESOLUTIONS[cfg['size']]
        width, height = res[0], res[1]
        
        test_picam2 = Picamera2()
        has_autofocus = is_autofocus_camera(test_picam2)

        timelapse_controls = {}
        exposure_val = str(cfg['timelapse_exposure']).lower()
        
        if exposure_val == "automatic":
            timelapse_controls = {"AeEnable": True, "AwbEnable": True}
        else:
            exposure_time_us = int(exposure_val) * 1000000
            timelapse_controls = {"AeEnable": False, "AwbEnable": False, "AnalogueGain": cfg['timelapse_gain'], "ExposureTime": exposure_time_us}
            if has_autofocus:
                timelapse_controls["AfMode"] = controls.AfModeEnum.Manual
                timelapse_controls["LensPosition"] = 0.0

        capture_format = "RGB888" if cfg['timelapse_color'] else "YUV420"
        still_config = test_picam2.create_still_configuration(main={"size": (width, height), "format": capture_format}, controls=timelapse_controls)
        test_picam2.configure(still_config)
        test_picam2.start()
        
        logging.info("[TEST] Attesa stabilizzazione camera per lo scatto di prova...")
        time.sleep(2)

        request = test_picam2.capture_request()
        captured_image = request.make_array('main')
        request.release()

        if cfg['timelapse_color']:
            final_frame = cv2.cvtColor(captured_image, cv2.COLOR_RGB2BGR)
        else:
            final_frame = cv2.cvtColor(captured_image, cv2.COLOR_YUV2GRAY_I420)
        
        output_dir = cfg['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, "TEST_PHOTO.jpg")
        cv2.imwrite(filename, final_frame)
        
        logging.info(f"[TEST] {Fore.GREEN}Foto di test salvata con successo in: {filename}{Style.RESET_ALL}")

    except Exception as e:
        logging.error(f"[TEST] Si è verificato un errore durante il test foto: {e}")
    finally:
        if test_picam2 and test_picam2.started:
            test_picam2.stop()
        if test_picam2:
            test_picam2.close()
            logging.info("[TEST] Camera di test chiusa correttamente.")
            
    input("\nPremi Invio per tornare al menu principale...")

def intervals_overlap(start1_str, end1_str, start2_str, end2_str):
    """Controlla se due intervalli di tempo si sovrappongono, gestendo la mezzanotte."""
    if not all([start1_str, end1_str, start2_str, end2_str]):
        return False

    # Converte le stringhe in oggetti time
    start1 = datetime.strptime(start1_str, "%H:%M").time()
    end1 = datetime.strptime(end1_str, "%H:%M").time()
    start2 = datetime.strptime(start2_str, "%H:%M").time()
    end2 = datetime.strptime(end2_str, "%H:%M").time()

    # Normalizza gli intervalli che passano la mezzanotte in due parti
    def get_intervals(start, end):
        if start < end:
            return [(start, end)]
        else:
            return [(start, datetime.strptime("23:59", "%H:%M").time()), (datetime.strptime("00:00", "%H:%M").time(), end)]

    intervals1 = get_intervals(start1, end1)
    intervals2 = get_intervals(start2, end2)

    # Controlla la sovrapposizione tra ogni coppia di sotto-intervalli
    for s1, e1 in intervals1:
        for s2, e2 in intervals2:
            # Formula classica per la sovrapposizione di intervalli
            if s1 < e2 and s2 < e1:
                return True
    return False

def validate_config(cfg):
    """
    Esegue una serie di controlli di validazione sulla configurazione.
    Restituisce (True, []) se valido, altrimenti (False, [lista_errori]).
    """
    errors = []
    
    # 1. Controlla la coerenza degli orari di scheduling
    if (cfg['meteor_start_time'] and not cfg['meteor_end_time']) or (not cfg['meteor_start_time'] and cfg['meteor_end_time']):
        errors.append("Errore: Per il Meteor Finder, devono essere specificati sia l'orario di inizio che di fine.")
    if (cfg['timelapse_start_time'] and not cfg['timelapse_end_time']) or (not cfg['timelapse_start_time'] and cfg['timelapse_end_time']):
        errors.append("Errore: Per il Timelapse, devono essere specificati sia l'orario di inizio che di fine.")

    # 2. Controlla la sovrapposizione degli orari
    if intervals_overlap(cfg['meteor_start_time'], cfg['meteor_end_time'], cfg['timelapse_start_time'], cfg['timelapse_end_time']):
        errors.append("Errore: Le pianificazioni per Meteor Finder e Timelapse si sovrappongono. Devono essere in orari distinti.")
        
    # 3. Controlla la logica dell'intervallo del timelapse
    exposure_val = str(cfg['timelapse_exposure']).lower()
    if exposure_val != "automatic":
        try:
            exposure_sec = int(exposure_val)
            if cfg['timelapse_interval'] <= exposure_sec:
                errors.append(f"Errore: L'intervallo del Timelapse ({cfg['timelapse_interval']}s) deve essere maggiore del tempo di esposizione ({exposure_sec}s).")
        except (ValueError, TypeError):
            errors.append(f"Errore: Il valore dell'esposizione del Timelapse ('{exposure_val}') non è un numero valido.")

    # 4. Controlla i permessi per lo spegnimento programmato
    if cfg.get('shutdown_time'):
        if os.geteuid() != 0:
            errors.append("Errore: Lo spegnimento programmato richiede permessi root.")

    # 5. Controlla che `downscale_factor` sia un intero positivo.
    try:
        if int(cfg.get('downscale_factor', 1)) <= 0:
            errors.append("Errore: 'downscale_factor' deve essere un numero intero maggiore di zero.")
    except (ValueError, TypeError):
        errors.append("Errore: 'downscale_factor' deve essere un numero intero.")

    if errors:
        return False, errors
    return True, []

def run_application():
    """
    Questa è la funzione principale che avvia la state machine e tutti i thread.
    Viene chiamata solo dopo che la configurazione è stata finalizzata.
    """
    global running, current_state, out, ffmpeg_proc, recording_event, reference_frame
    global last_event_time, record_start_time
    global frames_captured, frames_processed, frames_written, events_triggered, lock_perf, state_lock
    global picam2, frame_queue, output_queue, snapshot_queue, timelapse_writer_queue
    global STRATEGY, RESOLUTIONS, METEOR_EXPOSURE_LIMITS, HAS_AUTOFOCUS, fourcc, extension

    # === LOGGING SETUP (Now safe to call) ===
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if APP_CONFIG['no_log_events']: logger.setLevel(logging.CRITICAL + 1)
    for handler in logger.handlers[:]: logger.removeHandler(handler)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler("detection_log.log")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    console_formatter = ColoredFormatter("%(asctime)s [%(levelname)s] %(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # === PARAMETER SETUP ===
    STRATEGY = APP_CONFIG['strategy']
#    RESOLUTIONS = {"small": (640, 480), "medium": (1280, 720), "large": (1920, 1080)}
#    METEOR_EXPOSURE_LIMITS = (50000, 1000000)

    # Code di comunicazione thread-safe tra i vari componenti.
    frame_queue = queue.Queue(maxsize=APP_CONFIG['frame_queue_maxsize'])
    output_queue = queue.Queue(maxsize=APP_CONFIG['output_queue_maxsize'])
    snapshot_queue = queue.Queue(maxsize=APP_CONFIG['snapshot_queue_maxsize'])
    timelapse_writer_queue = queue.Queue(maxsize=10)

    # === CAMERA SETUP ===
    os.makedirs(APP_CONFIG['output_dir'], exist_ok=True)
    picam2 = Picamera2()
    HAS_AUTOFOCUS = is_autofocus_camera(picam2)
    width, height = RESOLUTIONS[APP_CONFIG['size']]

    # === CODEC ===
    # Determina l'estensione del file e il FourCC per i codec non-H264.
    if APP_CONFIG['codec'] == "avi":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        extension = "avi"
    elif APP_CONFIG['codec'] == "mjpeg":
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        extension = "avi"
    elif APP_CONFIG['codec'] == "h264":
        fourcc = None
        extension = "mp4"
    else:
        raise ValueError("Codec non supportato.")

    # === MAIN STATE MACHINE ===
    active_threads = []
    state_events = {'meteor_finder': threading.Event(), 'timelapse': threading.Event()}

    # Initialize global variables for threads
    reference_frame = None
    recording_event = threading.Event()
    out, ffmpeg_proc = None, None
    running = True
    last_event_time, record_start_time = 0, 0
    frames_captured, frames_processed, frames_written, events_triggered = 0, 0, 0, 0
    lock_perf = threading.Lock()
    state_lock = threading.Lock()

    with state_lock:
        current_state = "IDLE"

    monitor_t = threading.Thread(target=monitor_thread, daemon=True, name="Monitor")
    monitor_t.start()

    shutdown_time_obj = None
    if APP_CONFIG['shutdown_time']:
        try:
            shutdown_time_obj = datetime.strptime(APP_CONFIG['shutdown_time'], "%H:%M").time()
            logging.info(f"[MANAGER] Spegnimento del sistema programmato per le {APP_CONFIG['shutdown_time']}")
        except (ValueError, TypeError):
            logging.error(f"[MANAGER] Formato orario di spegnimento non valido.")

    shutdown_initiated = False
    shutdown_for_system = False

    try:
        while running:
            with state_lock:
                local_state = current_state

            desired_state = "IDLE"
            is_meteor_time = is_time_in_interval(APP_CONFIG['meteor_start_time'], APP_CONFIG['meteor_end_time'])
            is_timelapse_time = is_time_in_interval(APP_CONFIG['timelapse_start_time'], APP_CONFIG['timelapse_end_time'])
            if is_meteor_time:
                desired_state = "METEOR_FINDER"
            elif is_timelapse_time:
                desired_state = "TIMELAPSE"
            elif local_state == "TIMELAPSE" and not is_timelapse_time and APP_CONFIG['timelapse_to_video']:
                desired_state = "POST_PROCESSING"

            if desired_state != local_state:
                logging.info(f"[MANAGER] Transizione di stato: {local_state} -> {desired_state}", extra={'state': desired_state})
                if local_state != "IDLE" and local_state != "POST_PROCESSING":
                    logging.info(f"[MANAGER] Arresto della modalità {local_state}...")
                    state_events[local_state.lower()].clear()
                    for t in active_threads: t.join(timeout=15)
                    
                    alive_threads = [t for t in active_threads if t.is_alive()]
                    if alive_threads:
                        logging.warning(f"[MANAGER] I seguenti thread non sono terminati: {[t.name for t in alive_threads]}")
                    
                    active_threads.clear()
                    if picam2.started:
                        picam2.stop()
                    logging.info(f"[MANAGER] Modalità {local_state} arrestata.")

                with state_lock:
                    current_state = desired_state

                if desired_state == "METEOR_FINDER":
                    logging.info("[MANAGER] Riconfigurazione camera per METEOR_FINDER...")
                    lores_w = width // APP_CONFIG['downscale_factor']
                    lores_h = height // APP_CONFIG['downscale_factor']
                    meteor_controls = {"FrameDurationLimits": METEOR_EXPOSURE_LIMITS, "AnalogueGain": APP_CONFIG['gain']}
                    if APP_CONFIG['framerate_mode'] == 'fixed':
                        meteor_controls["FrameRate"] = APP_CONFIG['framerate']
                    video_config = picam2.create_video_configuration(
                        main={"size": (width, height), "format": "YUV420"},
                        lores={"size": (lores_w, lores_h), "format": "YUV420"},
                        controls=meteor_controls
                    )
                    picam2.configure(video_config)
                    picam2.start()
                    effective_framerate = APP_CONFIG['framerate']
                    if APP_CONFIG['framerate_mode'] == 'dynamic':
                        metadata = picam2.capture_metadata()
                        effective_framerate = metadata.get("FrameRate", APP_CONFIG['framerate'])
                    logging.info(f"[CAMERA] Framerate effettivo impostato a: {effective_framerate:.2f} fps")
                    pre_event_buffer = deque(maxlen=int(APP_CONFIG['pre_event_seconds'] * effective_framerate))
                    active_threads = [                           
                        threading.Thread(target=capture_thread_meteor, args=(state_events['meteor_finder'], width, height), daemon=True, name="MeteorCapture"),
                        threading.Thread(target=processing_thread, args=(state_events['meteor_finder'], effective_framerate, pre_event_buffer, width, height), daemon=True, name="MeteorProcess"),
                        threading.Thread(target=writer_thread, args=(state_events['meteor_finder'],), daemon=True, name="MeteorWriter"),
                        threading.Thread(target=snapshot_writer_thread, args=(state_events['meteor_finder'],), daemon=True, name="MeteorSnapshot")
                    ]
                    state_events['meteor_finder'].set()
                    for t in active_threads: t.start()

                elif desired_state == "TIMELAPSE":
                    logging.info("[MANAGER] Riconfigurazione camera per TIMELAPSE...")
                    timelapse_controls = {}
                    exposure_val = str(APP_CONFIG['timelapse_exposure']).lower()
                    if exposure_val == "automatic":
                        logging.info("[MANAGER] Modalità Timelapse: ESPOSIZIONE AUTOMATICA")
                        timelapse_controls = {"AeEnable": True, "AwbEnable": True}
                        capture_format = "RGB888" if APP_CONFIG['timelapse_color'] else "YUV420"
                    else:
                        logging.info(f"[MANAGER] Modalità Timelapse: ESPOSIZIONE MANUALE ({exposure_val}s)")
                        try:
                            exposure_time_us = int(exposure_val) * 1000000
                            timelapse_controls = {"AeEnable": False, "AwbEnable": False, "AnalogueGain": APP_CONFIG['timelapse_gain'], "ExposureTime": exposure_time_us}
                            if HAS_AUTOFOCUS:
                                timelapse_controls["AfMode"] = controls.AfModeEnum.Manual
                                timelapse_controls["LensPosition"] = 0.0
                            capture_format = "RGB888" if APP_CONFIG['timelapse_color'] else "YUV420"
                        except (ValueError, TypeError):
                            logging.error(f"[MANAGER] Valore di esposizione non valido: '{exposure_val}'. Impossibile avviare.")
                            with state_lock:
                                current_state = "IDLE"
                            continue
                    still_config = picam2.create_still_configuration(main={"size": (width, height), "format": capture_format}, controls=timelapse_controls)
                    picam2.configure(still_config)
                    picam2.start()
                    active_threads = [
                        threading.Thread(target=timelapse_capture_thread, args=(state_events['timelapse'],), daemon=True, name="TimelapseCapture"),
                        threading.Thread(target=timelapse_writer_thread, args=(state_events['timelapse'],), daemon=True, name="TimelapseWriter")
                    ]
                    state_events['timelapse'].set()
                    for t in active_threads: t.start()

                elif desired_state == "POST_PROCESSING":
                    logging.info("[MANAGER] Ingresso in modalità POST_PROCESSING...", extra={'state': 'POST_PROCESSING'})
                    timelapse_dir = os.path.join(APP_CONFIG['output_dir'], "timelapse")
                    video_filename = os.path.join(APP_CONFIG['output_dir'], f"timelapse_{datetime.now().strftime('%Y%m%d')}.mp4")
                    create_timelapse_video(image_folder=timelapse_dir, output_filename=video_filename, fps=APP_CONFIG['timelapse_video_fps'], cleanup=APP_CONFIG['timelapse_cleanup_images'])
                    with state_lock:
                        current_state = "IDLE"
                    logging.info("[MANAGER] Post-processing completato. Ritorno in modalità IDLE.", extra={'state': 'IDLE'})

                elif desired_state == "IDLE":
                    logging.info("[MANAGER] Ingresso in modalità IDLE.", extra={'state': 'IDLE'})

            if shutdown_time_obj and not shutdown_initiated:
                if datetime.now().time() >= shutdown_time_obj:
                    logging.info(f"[MANAGER] Orario di spegnimento raggiunto. Avvio della terminazione.", extra={'state': 'SHUTDOWN'})
                    shutdown_initiated = True
                    shutdown_for_system = True
                    running = False

            time.sleep(30)

    except KeyboardInterrupt:
        logging.info("[MAIN] Terminazione richiesta dall'utente...")
    finally:
        running = False
        for event in state_events.values(): event.clear()
        if 'recording_event' in globals() and recording_event.is_set():
            logging.info("[MAIN] Finalizzazione della registrazione in corso...")
            output_queue.put(None)
        all_threads = [monitor_t] + active_threads
        for t in all_threads: t.join(timeout=5)
        if 'picam2' in globals() and picam2.started: picam2.stop()
        if 'out' in globals() and out: out.release()
        if 'ffmpeg_proc' in globals() and ffmpeg_proc:
            if ffmpeg_proc.stdin:
                try: ffmpeg_proc.stdin.close()
                except BrokenPipeError: pass
            ffmpeg_proc.wait()
        logging.info("[MAIN] Uscita completata.")

    if shutdown_for_system:
        logging.info("[SHUTDOWN] Esecuzione del comando di spegnimento del sistema...", extra={'state': 'SHUTDOWN'})
        os.sync()
        time.sleep(2)
        os.system("sudo shutdown now")


# === MAIN: The State Machine Manager ===
if __name__ == "__main__":
    """
    SEZIONE CRITICA: Il ciclo di controllo principale dell'applicazione.
    - Agisce come uno scheduler, controllando l'ora per determinare lo stato desiderato.
    - Gestisce il ciclo di vita di tutti i thread (avvio, arresto, join).
    - Esegue le transizioni di stato in modo sicuro, arrestando i vecchi thread
      e riconfigurando la camera prima di avviare i nuovi.
    - Gestisce lo spegnimento programmato e la chiusura pulita (`KeyboardInterrupt`).
    """
    clear_screen()
    init()
    
    # 1. Definisci i valori predefiniti di base.
    master_defaults = {
        "size": "medium", "binning": True, "gain": 8.0, "codec": "h264", "output_dir": "output",
        "framerate_mode": "fixed", "framerate": 30 if pi_model == "pi5" else 15,
        "meteor_start_time": "22:00", "meteor_end_time": "05:00",
        "strategy": "contour", "record_duration": 12, "pre_event_seconds": 5,
        "min_brightness": 50, "min_area": 5,
        "diff_threshold": 25, "trigger_area": 4000, "learning_rate": 0.01,
        "timelapse_start_time": None, "timelapse_end_time": None,
        "timelapse_exposure": "30", "timelapse_interval": 35,
        "timelapse_gain": 8.0, "timelapse_color": False,
        "timelapse_to_video": False, "timelapse_video_fps": 24, "timelapse_cleanup_images": False,
        "no_log_events": False, "shutdown_time": None,
        "frame_queue_maxsize": 20, "output_queue_maxsize": 120,
        "snapshot_queue_maxsize": 10, "downscale_factor": 4
    }


    # 2. Carica la configurazione da file o crea un nuovo file.
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            try:
                config_data = json.load(f)
            except Exception as e:
                print(f"{Fore.RED}[WARNING] Impossibile analizzare {CONFIG_FILE}: {e}. Verranno usati i predefiniti.{Style.RESET_ALL}")
                config_data = master_defaults
    else:
        print(f"[CONFIG] File di configurazione non trovato. Creazione di {CONFIG_FILE} con i valori predefiniti.")
        save_config(master_defaults)
        config_data = master_defaults

    # --- Loop di Configurazione e Validazione ---
    while True:
        # Avvia l'editor interattivo.
        final_config = edit_config_interactive(config_data)

        # Se l'utente esce dall'editor, termina il programma.
        if final_config is None:
            print("\nUscita dal programma.")
            sys.exit(0)

        # Valida la configurazione proposta dall'utente.
        is_valid, errors = validate_config(final_config)

        if is_valid:
            # Se la configurazione è valida, esci dal loop e avvia l'applicazione.
            APP_CONFIG = final_config
            print("\n" + "="*15 + " AVVIO APPLICAZIONE " + "="*15)
            run_application()
            break # Esce dal while True
        else:
            # Se ci sono errori, mostrali e torna all'editor.
            print(f"\n{Fore.RED}{Style.BRIGHT}ATTENZIONE: Sono stati rilevati errori di configurazione:{Style.RESET_ALL}")
            for error in errors:
                print(f"{Fore.RED}  - {error}{Style.RESET_ALL}")
            input(f"\n{Style.DIM}Premi Invio per tornare all'editor e correggere...{Style.RESET_ALL}")
            # Ricarica l'editor con l'ultima configurazione (errata) per la correzione.
            config_data = final_config

    # 3. Avvia l'editor interattivo.
    final_config = edit_config_interactive(config_data)

    # 4. Se l'utente ha scelto di avviare, popola la configurazione globale e avvia l'applicazione.
    if final_config:
        APP_CONFIG = final_config
        print("\n" + "="*15 + " AVVIO APPLICAZIONE " + "="*15)
        run_application()
    else:
        print("Avvio annullato dall'utente.")
