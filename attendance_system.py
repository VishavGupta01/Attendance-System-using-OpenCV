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

# Load configuration settings
def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    global KNOWN_FACES_DIR, ATTENDANCE_CSV_PATH
    # defaults are set in case config fails
    KNOWN_FACES_DIR = config.get('Paths', 'KNOWN_FACES_DIR', fallback='known_faces')
    ATTENDANCE_CSV_PATH = config.get('Paths', 'ATTENDANCE_CSV_PATH', fallback='attendance.csv')

load_config()

# List to store known face encodings and names
known_face_encodings = []
known_face_names = []

# Set to track attendance in the current session
attendance_set = set()

# Load known faces and their encodings
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
                # Load an image file
                image_path = os.path.join(KNOWN_FACES_DIR, filename)
                image = face_recognition.load_image_file(image_path)

                # Encode the face
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    face_encoding = encodings[0]
                    # Store the encoding and the name
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(os.path.splitext(filename)[0])
                else:
                    print(f"Warning: No face found in {filename}")

        update_status(f"Loaded {len(known_face_names)} faces successfully.", "green")
    except Exception as e:
        update_status(f"Error loading faces: {e}", "red")

# Record attendance in a CSV file if not already recorded in the current session
def mark_attendance(name):
    if name not in attendance_set:
        try:
            # Check if file exists to determine if we need a header
            file_exists = os.path.exists(ATTENDANCE_CSV_PATH)

            with open(ATTENDANCE_CSV_PATH, 'a') as f:
                if not file_exists:
                    f.write('Name,DateTime\n')  # Header for the CSV file

                now = datetime.now()
                dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
                f.write(f'{name},{dt_string}\n')

            attendance_set.add(name)
            attendance_log.insert(END, f'{name} marked present at {dt_string}\n')
            attendance_log.see(END)
            update_status(f"Attendance marked for {name}.", "green")
        except Exception as e:
            update_status(f"Error marking attendance: {e}", "red")

# Start the camera and process frames
def start_camera(camera_index=0):
    load_known_faces()

    # Try to read camera index from config, otherwise use 0
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

    # Initialize variables for optimizing frame rate
    process_this_frame = True
    face_locations = []
    face_names = []

    while camera_running:
        ret, frame = video_capture.read()
        if not ret:
            update_status("Failed to capture video.", "red")
            break

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        # We ensure the array is contiguous in memory for dlib
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # Or instead, use the known face with the smallest distance to the new face
                if len(known_face_encodings) > 0:
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                face_names.append(name)

                # Mark attendance if the face is recognized
                if name != "Unknown":
                    mark_attendance(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Display the name
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert the frame to ImageTk format for display in tkinter
        # Also ensure contiguous here just to be safe for PIL
        display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Note: Updating GUI from a thread is technically not thread-safe in Tkinter,
        # but for simple apps it often works. If it crashes later, use root.after().
        try:
            display_frame.imgtk = imgtk
            display_frame.configure(image=imgtk)
            root.update()
        except Exception:
            break

    video_capture.release()
    # cv2.destroyAllWindows() # Not needed as we are using Tkinter

# Stop the camera
def stop_camera():
    global camera_running
    camera_running = False
    update_status("Camera stopped.", "blue")

# Start a new session
def start_new_session():
    global attendance_set
    attendance_set.clear()
    attendance_log.delete(1.0, END)
    update_status("New session started.", "green")

# Update status in the GUI
def update_status(message, color="black"):
    if 'status_label' in globals() and status_label.winfo_exists():
        status_label.config(text=f"Status: {message}", fg=color)

# Authenticate user before starting the application
def authenticate_user():
    # Try to read password from config
    config = configparser.ConfigParser()
    config.read('config.ini')
    correct_password = config.get('Security', 'PASSWORD', fallback='admin')

    user_input = input("Enter password: ")
    # Strip whitespace just in case
    if user_input.strip() != correct_password.strip():
        print("Access denied")
        exit()
    print("Access granted")

# Main function to initialize the GUI
def main():
    authenticate_user()

    global root, display_frame, attendance_log, status_label, camera_running
    root = tk.Tk()
    root.title("Smart Attendance System")
    root.geometry("800x600")

    display_frame = Label(root)
    display_frame.pack(pady=10)

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=5)

    start_button = Button(btn_frame, text="Start Camera", command=lambda: threading.Thread(target=start_camera_thread, daemon=True).start(), bg="green", fg="white")
    start_button.pack(side=tk.LEFT, padx=5)

    stop_button = Button(btn_frame, text="Stop Camera", command=stop_camera, bg="red", fg="white")
    stop_button.pack(side=tk.LEFT, padx=5)

    new_session_button = Button(btn_frame, text="New Session", command=start_new_session, bg="blue", fg="white")
    new_session_button.pack(side=tk.LEFT, padx=5)

    status_label = Label(root, text="Status: Waiting for action", fg="blue")
    status_label.pack(pady=5)

    log_frame = tk.Frame(root)
    log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    scrollbar = Scrollbar(log_frame, orient=VERTICAL)
    attendance_log = Text(log_frame, height=10, yscrollcommand=scrollbar.set)
    scrollbar.config(command=attendance_log.yview)

    scrollbar.pack(side=RIGHT, fill=Y)
    attendance_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    camera_running = False

    root.mainloop()

# Start camera thread
def start_camera_thread():
    global camera_running
    if camera_running:
        return # Prevent multiple threads
    camera_running = True
    start_camera()

if __name__ == "__main__":
    main()