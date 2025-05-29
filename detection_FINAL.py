import time
import torch
import cv2
import os
import csv
import re

from ultralytics import YOLO
from easyocr import Reader
from datetime import datetime

# Threshold confidence untuk deteksi plat nomor
CONFIDENCE_THRESHOLD = 0.7
# Warna hijau untuk bounding box
COLOR = (0, 255, 0)

def detect_number_plates(image, model, display=False):
    """
    Mendeteksi plat nomor pada gambar menggunakan model YOLO.
    Args:
        image: Gambar input (numpy array)
        model: Model YOLO untuk deteksi plat
        display: Flag untuk menampilkan gambar plat yang terdeteksi (default: False)
    Returns:
        List berisi bounding box plat nomor yang terdeteksi
    """
    start = time.time()
    # Lakukan prediksi menggunakan model
    detections = model.predict(image)[0].boxes.data

    # Periksa apakah ada deteksi
    if detections.shape != torch.Size([0, 6]):

        # Inisialisasi list untuk bounding box dan confidence
        boxes = []
        confidences = []

        # Loop pada setiap deteksi
        for detection in detections:
            confidence = detection[4]

            # Filter deteksi dengan confidence di bawah threshold
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            # Tambahkan bounding box dan confidence ke list
            boxes.append(detection[:4])
            confidences.append(detection[4])

        print(f"{len(boxes)} Number plate(s) have been detected.")
        number_plate_list= []

        # Loop pada setiap bounding box
        for i in range(len(boxes)):
            # Ekstrak koordinat bounding box
            xmin, ymin, xmax, ymax = int(boxes[i][0]), int(boxes[i][1]),\
                                    int(boxes[i][2]), int(boxes[i][3])
            # Tambahkan bounding box ke list hasil
            number_plate_list.append([[xmin, ymin, xmax, ymax]])

            # Gambar bounding box dan label pada gambar
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), COLOR, 2)
            text = "Number Plate: {:.2f}%".format(confidences[i] * 100)
            cv2.putText(image, text, (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)

            # Jika display True, tampilkan gambar plat yang dipotong
            if display:
                number_plate = image[ymin:ymax, xmin:xmax]
                cv2.imshow(f"Number plate {i}", number_plate)

        end = time.time()
        print(f"Time to detect the number plates: {(end - start) * 1000:.0f} milliseconds")
        return number_plate_list
    else:
        print("No number plates have been detected.")
        return []

def recognize_number_plates(image_or_path, reader, number_plate_list, write_to_csv=False):
    """
    Mengenali teks pada plat nomor yang terdeteksi menggunakan EasyOCR.
    Args:
        image_or_path: Gambar (numpy array) atau path ke gambar
        reader: Reader EasyOCR
        number_plate_list: List berisi bounding box plat nomor
        write_to_csv: Flag untuk menyimpan hasil ke CSV (default: False)
    Returns:
        List berisi bounding box dan teks plat yang dikenali
    """
    start = time.time()
    # Jika input adalah path, baca gambar. Jika numpy array, gunakan langsung.
    image = cv2.imread(image_or_path) if isinstance(image_or_path, str)\
                                        else image_or_path

    # Loop pada setiap bounding box plat nomor
    for i, box in enumerate(number_plate_list):
        # Potong area plat nomor
        np_image = image[box[0][1]:box[0][3], box[0][0]:box[0][2]]

        # Gunakan EasyOCR untuk membaca teks
        detection = reader.readtext(np_image, paragraph=True)

        if len(detection) == 0:
            # Jika tidak ada teks terdeteksi
            text = ""
        else:
            # Ambil teks terdeteksi
            text = str(detection[0][1])

        # Tambahkan teks ke dalam list
        number_plate_list[i].append(text)

    # Jika write_to_csv True, simpan hasil ke file CSV
    if write_to_csv:
        csv_file = open("number_plates.csv", "w")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["image_path", "box", "text"])

        for box, text in number_plate_list:
            csv_writer.writerow([image_or_path, box, text])
        csv_file.close()

    end = time.time()
    print(f"Time to recognize the number plates: {(end - start) * 1000:.0f} milliseconds")
    return number_plate_list
    
def extract_tax_info(text):
    """
    Ekstrak informasi bulan dan tahun pajak dari teks plat nomor.
    Format yang diharapkan: dua digit terakhir tahun (contoh: 23) 
    dan bulan (contoh: 07) yang berdekatan.
    Args:
        text: Teks plat nomor yang telah dikenali
    Returns:
        tax_month: Bulan pajak (string dua digit)
        tax_year: Tahun pajak (string dua digit)
    """
    # Cari semua urutan digit (2-4 digit) dalam teks
    matches = re.findall(r'\d{2,4}', text)
    
    if len(matches) >= 2:
        # Ambil dua digit terakhir dari match terakhir sebagai tahun
        tax_year = matches[-1][-2:]
        
        # Ambil dua digit sebagai bulan:
        # - Jika match terakhir memiliki 4 digit, ambil dua digit pertama
        # - Jika tidak, ambil dua digit terakhir dari match kedua terakhir
        if len(matches[-1]) >= 4:
            tax_month = matches[-1][:2]
        elif len(matches) >= 2:
            tax_month = matches[-2][-2:]
        else:
            return None, None
        
        return tax_month, tax_year
    return None, None

def validate_tax(tax_month, tax_year):
    """
    Validasi status pajak berdasarkan bulan dan tahun.
    Args:
        tax_month: Bulan pajak (string dua digit)
        tax_year: Tahun pajak (string dua digit)
    Returns:
        status: Status pajak ("AKTIF", "KADALUARSA", atau pesan error)
        masa_berlaku: String yang menggambarkan masa berlaku pajak
    """
    if not tax_month or not tax_year:
        return "Tidak Diketahui", "Informasi tidak lengkap"
    
    try:
        # Konversi ke integer
        month = int(tax_month)
        year = int(tax_year) + 2000  # Asumsi tahun 2000+
        
        # Validasi rentang bulan (1-12)
        if month < 1 or month > 12:
            return "Invalid", "Bulan tidak valid"
        
        # Hitung tanggal akhir masa pajak (akhir bulan)
        if month == 12:
            next_month = 1
            next_year = year + 1
        else:
            next_month = month + 1
            next_year = year
        
        # Tanggal akhir masa pajak adalah hari pertama bulan berikutnya
        tax_end_date = datetime(next_year, next_month, 1)
        current_date = datetime.now()
        
        # Tentukan status: AKTIF jika sekarang belum melewati tax_end_date
        status = "AKTIF" if current_date <= tax_end_date else "KADALUARSA"
        
        # Daftar nama bulan untuk tampilan
        nama_bulan = [
            "Januari", "Februari", "Maret", "April", "Mei", "Juni",
            "Juli", "Agustus", "September", "Okttober", "November", "Desember"
        ][month-1]
        
        masa_berlaku = f"{nama_bulan} {year}"
        return status, masa_berlaku
    
    except:
        # Jika terjadi error, kembalikan pesan error
        return "Error", "Terjadi kesalahan"

# Jika script dijalankan langsung (bukan diimpor)
if __name__ == "__main__":
    # Load model YOLO dari direktori lokal
    model = YOLO("dataset/runs/detect/car_plate_detection4/weights/best.pt")
    # Inisialisasi reader EasyOCR untuk bahasa Inggris, menggunakan GPU
    reader = Reader(['en'], gpu=True)

    # Path file yang akan diproses
    file_path = "gambar_sendiri_4.jpg"
    # Ekstrak ekstensi file
    _, file_extension = os.path.splitext(file_path)

    # Proses gambar
    if file_extension in ['.jpg', '.jpeg', '.png']:
        print("Processing the image...")
        image = cv2.imread(file_path)
        # Deteksi plat nomor
        number_plate_list = detect_number_plates(image, model, display=True)
        cv2.imshow('Image', image)
        cv2.waitKey(0)

        # Jika ada plat terdeteksi, kenali teksnya
        if number_plate_list != []:
            number_plate_list = recognize_number_plates(file_path, reader,
                                                        number_plate_list,
                                                        write_to_csv=True)
            # Tampilkan teks pada gambar
            for box, text in number_plate_list:
                cv2.putText(image, text, (box[0], box[3] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)
            cv2.imshow('Image', image)
            cv2.waitKey(0)

    # Proses video
    elif file_extension in ['.mp4', '.mkv', '.avi', '.wmv', '.mov']:
        print("Processing the video...")
        # Buka video
        video_cap = cv2.VideoCapture(file_path)
        # Dapatkan properti video
        frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_cap.get(cv2.CAP_PROP_FPS))
        # Inisialisasi video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("output.mp4", fourcc, fps,
                                (frame_width, frame_height))

        # Loop frame video
        while True:
            start = time.time()
            success, frame = video_cap.read()

            # Jika tidak ada frame lagi, keluar dari loop
            if not success:
                print("There are no more frames to process."
                        " Exiting the script...")
                break

            # Deteksi plat nomor pada frame
            number_plate_list = detect_number_plates(frame, model)

            # Jika ada plat terdeteksi, kenali teksnya
            if number_plate_list != []:
                number_plate_list = recognize_number_plates(frame, reader,
                                                        number_plate_list)

                # Tampilkan teks pada frame
                for box, text in number_plate_list:
                    cv2.putText(frame, text, (box[0], box[3] + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR, 2)

            # Hitung dan tampilkan FPS
            end = time.time()
            fps_text = f"FPS: {1 / (end - start):.2f}"
            cv2.putText(frame, fps_text, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)

            # Tampilkan frame
            cv2.imshow("Output", frame)
            # Tulis frame ke output video
            writer.write(frame)
            # Keluar jika tombol 'q' ditekan
            if cv2.waitKey(10) == ord("q"):
                break

        # Rilis sumber daya
        video_cap.release()
        writer.release()
        cv2.destroyAllWindows()