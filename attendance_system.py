import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
import tkinter as tk
from tkinter import Label, Button, Text, Scrollbar, VERTICAL, RIGHT, Y, END
from PIL import Image, ImageTk
import threading
import configparser

def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    global KNOWN_FACES_DIR, ATTENDANCE_CSV_PATH
    KNOWN_FACES_DIR = config.get('Paths', 'KNOWN_FACES_DIR', fallback='known_faces')
    ATTENDANCE_CSV_PATH = config.get('Paths', 'ATTENDANCE_CSV_PATH', fallback='attendance.csv')

load_config()

known_face_encodings = []
known_face_names = []

attendance_set = set()

def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []

    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        update_status(f"Created missing directory: {KNOWN_FACES_DIR}", "blue")
        return

    try:
        files = os.listdir(KNOWN_FACES_DIR)
        if not files:
            update_status("No images found in known_faces folder.", "blue")
            return

        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(KNOWN_FACES_DIR, filename)
                image = face_recognition.load_image_file(image_path)

                encodings = face_recognition.face_encodings(image)

                if encodings:
                    face_encoding = encodings[0]
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(os.path.splitext(filename)[0])
                else:
                    print(f"Warning: No face found in {filename}")

        update_status(f"Loaded {len(known_face_names)} faces successfully.", "green")
    except Exception as e:
        update_status(f"Error loading faces: {e}", "red")

def mark_attendance(name):
    if name not in attendance_set:
        try:
            file_exists = os.path.exists(ATTENDANCE_CSV_PATH)

            with open(ATTENDANCE_CSV_PATH, 'a') as f:
                if not file_exists:
                    f.write('Name,DateTime\n')

                now = datetime.now()
                dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
                f.write(f'{name},{dt_string}\n')

            attendance_set.add(name)
            attendance_log.insert(END, f'{name} marked present at {dt_string}\n')
            attendance_log.see(END)
            update_status(f"Attendance marked for {name}.", "green")
        except Exception as e:
            update_status(f"Error marking attendance: {e}", "red")

def start_camera(camera_index=0):
    load_known_faces()

    config = configparser.ConfigParser()
    config.read('config.ini')
    try:
        idx = config.getint('Camera', 'CAMERA_INDEX', fallback=0)
    except:
        idx = 0

    video_capture = cv2.VideoCapture(idx)

    if not video_capture.isOpened():
        update_status("Could not open video device.", "red")
        return

    process_this_frame = True
    face_locations = []
    face_names = []

    while camera_running:
        ret, frame = video_capture.read()
        if not ret:
            update_status("Failed to capture video.", "red")
            break

        frame = cv2.resize(frame, (640, 480))

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                if len(known_face_encodings) > 0:
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                face_names.append(name)
                if name != "Unknown":
                    mark_attendance(name)

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        try:
            display_frame.imgtk = imgtk
            display_frame.configure(image=imgtk)
            root.update()
        except Exception:
            break

    video_capture.release()

def stop_camera():
    global camera_running
    camera_running = False

    display_frame.configure(image='', text="[ Camera Off ]")
    display_frame.image = None

    update_status("Camera stopped.", "#F1C40F")

def start_new_session():
    global attendance_set
    attendance_set.clear()
    attendance_log.delete(1.0, END)
    update_status("New session started.", "green")

def update_status(message, color="black"):
    if 'status_label' in globals() and status_label.winfo_exists():
        status_label.config(text=f"Status: {message}", fg=color)

def authenticate_user():
    config = configparser.ConfigParser()
    config.read('config.ini')
    correct_password = config.get('Security', 'PASSWORD', fallback='admin')

    user_input = input("Enter password: ")
    if user_input.strip() != correct_password.strip():
        print("Access denied")
        exit()
    print("Access granted")

def main():
    authenticate_user()

    global root, display_frame, attendance_log, status_label, camera_running

    COLOR_BG = "#2C3E50"
    COLOR_FG = "#ECF0F1"
    COLOR_ACCENT = "#34495E"
    COLOR_BTN_START = "#27AE60"
    COLOR_BTN_STOP = "#C0392B"
    COLOR_BTN_NEW = "#2980B9"

    root = tk.Tk()
    root.title("Smart Attendance System")
    root.geometry("900x750")
    root.configure(bg=COLOR_BG)

    header_frame = tk.Frame(root, bg=COLOR_ACCENT, pady=10)
    header_frame.pack(fill=tk.X)
    title_label = Label(header_frame, text="Smart Face Recognition Attendance",
                        font=("Helvetica", 18, "bold"), bg=COLOR_ACCENT, fg=COLOR_FG)
    title_label.pack()

    video_container = tk.Frame(root, width=640, height=480, bg="black",
                            highlightbackground=COLOR_FG, highlightthickness=2)

    video_container.pack_propagate(False)
    video_container.pack(pady=20)

    display_frame = Label(video_container, text="[ Camera Off ]",
                        font=("Helvetica", 14), bg="black", fg="white")
    display_frame.pack(fill=tk.BOTH, expand=True)

    btn_frame = tk.Frame(root, bg=COLOR_BG)
    btn_frame.pack(pady=10)

    btn_style = {"font": ("Helvetica", 12, "bold"), "width": 15, "pady": 5, "relief": "flat", "fg": "white"}

    start_button = Button(btn_frame, text="Start Camera",
                        command=lambda: threading.Thread(target=start_camera_thread, daemon=True).start(),
                          bg=COLOR_BTN_START, **btn_style)
    start_button.pack(side=tk.LEFT, padx=10)

    stop_button = Button(btn_frame, text="Stop Camera", command=stop_camera, bg=COLOR_BTN_STOP, **btn_style)
    stop_button.pack(side=tk.LEFT, padx=10)

    new_session_button = Button(btn_frame, text="New Session", command=start_new_session, bg=COLOR_BTN_NEW, **btn_style)
    new_session_button.pack(side=tk.LEFT, padx=10)

    status_label = Label(root, text="Status: Ready", font=("Helvetica", 10, "italic"), bg=COLOR_BG, fg="#F1C40F")
    status_label.pack(pady=5)

    log_frame = tk.Frame(root, bg=COLOR_BG)
    log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

    scrollbar = Scrollbar(log_frame, orient=VERTICAL)
    attendance_log = Text(log_frame, height=8, yscrollcommand=scrollbar.set,
                        font=("Consolas", 10), bg="#000000", fg="#00FF00", bd=0)

    scrollbar.config(command=attendance_log.yview)
    scrollbar.pack(side=RIGHT, fill=Y)
    attendance_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    camera_running = False
    root.mainloop()

def start_camera_thread():
    global camera_running
    if camera_running:
        return
    camera_running = True
    start_camera()

if __name__ == "__main__":
    main()