# --- START OF FILE stelle_cadenti_v3_final.py ---

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
    """Converte l'oggetto degli argomenti (Namespace) in un dizionario e lo salva come file JSON."""
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

# (Config file loading and argument parsing is unchanged)
# ...

STRATEGY = args.strategy

# === LOGGING ===
# ... (Unchanged)

# === PARAMETRI DERIVATI ===
RESOLUTIONS = {"small": (640, 480), "medium": (1280, 720), "large": (1920, 1080)}

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
    """Interroga le proprietà della camera per determinare se ha un autofocus (es. Camera Module 3)."""
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

# === H264 & CODEC SETUP ===
# ... (Unchanged)

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
    # ... (Logic is the same, but the main loop checks `state_event.is_set()`)

def writer_thread(state_event):
    # ... (Logic is the same, but the main loop checks `state_event.is_set()`)

def snapshot_writer_thread(state_event):
    # ... (Logic is the same, but the main loop checks `state_event.is_set()`)

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
                        controls={"FrameDurationLimits": (10000, 33333), "AnalogueGain": args.gain, "FrameRate": args.framerate}
                    )
                    picam2.configure(video_config)
                    picam2.start()
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