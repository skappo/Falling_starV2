# Falling Star V2 - Meteor and Timelapse Camera

This script is a comprehensive, multi-threaded application for a Raspberry Pi equipped with a PiCamera. It operates in two main modes: a high-framerate **Meteor Finder** for detecting and recording transient night-sky events, and a long-exposure **Timelapse** mode for creating star-trail videos.

The script is designed to be robust, running continuously and switching between modes based on a user-defined schedule. It also features a persistent configuration system, detailed logging, and optional post-processing capabilities.

## Dependencies

Before running the script, ensure the following dependencies are installed.

**Required Packages:**
*   OpenCV (`opencv-python`)
*   NumPy (`numpy`)
*   Picamera2 (`picamera2`)
*   Colorama (`colorama`)
*   FFmpeg

You can install the Python packages using `pip`:
```bash
pip install opencv-python numpy picamera2 colorama
```

On a Raspberry Pi, it is often recommended to install these packages using `apt` for better compatibility:
```bash
sudo apt update
sudo apt install python3-opencv python3-numpy python3-picamera2 ffmpeg
```

## How It Works

The script operates as a state machine that can be in one of four states: `IDLE`, `METEOR_FINDER`, `TIMELAPSE`, or `POST_PROCESSING`. It uses the system time to decide which mode should be active based on the start and end times you provide for each.

### Configuration

The script uses a `config.json` file to store settings.

*   When you run the script for the first time, it will create a `config.json` with default values.
*   You can edit this file directly to change the default behavior.
*   **Any arguments you provide on the command line will override the settings in `config.json` and will be saved back to the file for future runs.** This makes it easy to test a new setting temporarily and have it become the new default.

To reset to the original defaults, simply delete the `config.json` file.

## Basic Usage

To run the script with the default or saved configuration, simply execute:
```bash
python3 falling_starV2-1.py
```

To run both Meteor Finder and Timelapse mode on a schedule:
```bash
python3 falling_starV2-1.py --meteor_start_time "22:00" --meteor_end_time "05:00" --timelapse_start_time "19:00" --timelapse_end_time "21:30"
```

The script will automatically switch to the correct mode at the specified times. If no mode is active, it will enter an `IDLE` state to conserve resources.

## Operating Modes and Arguments

Here is a detailed breakdown of the arguments, grouped by the mode they affect.

---

### Meteor Finder Mode

This mode uses a high framerate and real-time image analysis to detect and record brief events like meteors.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--meteor_start_time` | HH:MM | `22:00` | The time to automatically start Meteor Finder mode. |
| `--meteor_end_time` | HH:MM | `05:00` | The time to automatically stop Meteor Finder mode. |
| `--strategy` | choice | `contour` | The detection algorithm to use. `contour` finds bright objects, while `diff` looks for changes between frames. |
| `--record_duration` | int | `10` | The duration (in seconds) to record for after an event is triggered. |
| `--pre_event_seconds`| int | `5` | How many seconds of video to save from *before* the event was triggered. |
| `--gain` | float | `8.0` | The analog gain for the camera sensor, increasing sensitivity. |
| `--framerate` | int | `30` (Pi 5) / `20` (Pi 4) | The target framerate for video capture. |

**Contour Strategy Settings:**
| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--min_brightness` | int | `50` | The minimum pixel brightness (0-255) to be considered part of a potential event. |
| `--min_area` | int | `5` | The minimum area (in pixels) an object must have to trigger a recording. |

**Diff Strategy Settings:**
| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--diff_threshold` | int | `25` | The amount a pixel's brightness must change (0-255) to be considered different from the background. |
| `--trigger_area` | int | `4000`| The number of pixels that must change to trigger a recording. |
| `--learning_rate` | float | `0.01` | How quickly the background reference frame adapts to slow changes (like clouds or moon). Lower is slower. |

---

### Timelapse Mode

This mode takes long-exposure still images at regular intervals. It can optionally compile these images into a video file when the session ends.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--timelapse_start_time` | HH:MM | `None` | The time to automatically start Timelapse mode. |
| `--timelapse_end_time` | HH:MM | `None` | The time to automatically stop Timelapse mode. |
| `--timelapse_exposure` | int | `15` | The exposure time in **seconds** for each photo. |
| `--timelapse_interval` | int | `20` | The total time in **seconds** between the start of one photo and the start of the next. Must be longer than exposure time. |
| `--timelapse_gain` | float | `8.0` | The analog gain (ISO) for the timelapse photos. |
| `--timelapse_color` | flag | `False` | If set, saves timelapse images in color. Default is monochrome. |
| `--timelapse_to_video` | flag | `False` | If set, automatically creates an `.mp4` video from the JPGs after the timelapse session ends. |
| `--timelapse_video_fps`| int | `24` | The framerate for the final compiled video. |
| `--timelapse_cleanup_images`| flag | `False` | If set, deletes the original JPG images after the video is created. **Use with caution!** |

---

### General & Performance Settings

These arguments control the overall behavior of the script across all modes.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--size` | choice | `medium` | Capture resolution: `small` (640x480), `medium` (1280x720), `large` (1920x1080). |
| `--output_dir` | str | `output`| The main directory where all videos and images will be saved. |
| `--codec` | choice | `h264` | The video codec for Meteor Finder events. `h264` uses hardware encoding on the Pi for best performance. |
| `--framerate_mode` | choice | `fixed` | `fixed` attempts to lock the framerate. `dynamic` allows the camera to adjust it based on lighting. |
| `--binning` | flag | `True` | Enables 2x2 sensor binning for the Meteor Finder, boosting light sensitivity at the cost of resolution. |
| `--downscale_factor` | int | `4` | Factor by which to downscale the image for motion detection analysis, improving performance. |
| `--shutdown_time` | HH:MM | `None` | If set, the script will execute `sudo shutdown now` at the specified time. Requires running with `sudo`. |
| `--no_log_events` | flag | `False`| Disables all logging to the console and file. |
| `--frame_queue_maxsize`| int | `30` | (Advanced) Max size of the internal queue for frames pending analysis. |
| `--output_queue_maxsize`| int | `60` | (Advanced) Max size of the internal queue for video frames pending writing to disk. |
| `--snapshot_queue_maxsize`| int| `10` | (Advanced) Max size of the internal queue for event snapshots pending writing to disk. |
