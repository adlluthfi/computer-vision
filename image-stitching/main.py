import cv2
import numpy as np
import os
from image_stitcher import ImageStitcher

def simple_stitch(images):
    """Metode simple menggunakan OpenCV Stitcher (lebih mudah)"""
    stitcher = cv2.Stitcher_create() if imutils.is_cv3(or_better=True) else cv2.createStitcher()
    (status, stitched) = stitcher.stitch(images)
    
    if status == 0:
        return stitched
    else:
        print(f"Stitching gagal dengan status: {status}")
        return None

def advanced_stitch(image_paths):
    """Metode advanced menggunakan custom stitcher"""
    stitcher = ImageStitcher()
    
    # Load gambar
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Gagal membaca: {path}")
            return None
        images.append(img)
    
    print(f"Berhasil load {len(images)} gambar")
    
    # Stitch gambar pertama dan kedua
    print("Menggabungkan gambar 1 dan 2...")
    result = stitcher.stitch([images[0], images[1]])
    
    if result is None:
        print("Gagal menggabungkan gambar 1 dan 2")
        return None
    
    # Stitch dengan gambar ketiga jika ada
    if len(images) > 2:
        print("Menggabungkan dengan gambar 3...")
        result = stitcher.stitch([result, images[2]])
        
        if result is None:
            print("Gagal menggabungkan gambar 3")
            return None
    
    return result

def crop_black_borders(image):
    """Crop border hitam hasil stitching"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(c)
        cropped = image[y:y+h, x:x+w]
        return cropped
    
    return image

if __name__ == "__main__":
    import imutils
    
    # Path ke gambar
    image_folder = "images/"
    image_paths = [
        os.path.join(image_folder, "gambar1.png"),
        os.path.join(image_folder, "gambar2.png"),
        os.path.join(image_folder, "gambar3.png")
    ]
    
    print("=== Image Stitching Program ===")
    print("\nPilih metode:")
    print("1. Simple Stitcher (OpenCV built-in)")
    print("2. Advanced Stitcher (Custom)")
    
    choice = input("\nPilih metode (1/2): ")
    
    # Cek file ada atau tidak
    for path in image_paths:
        if not os.path.exists(path):
            print(f"File tidak ditemukan: {path}")
            print(f"Pastikan file berada di folder '{image_folder}'")
            exit()
    
    result = None
    
    if choice == "1":
        print("\nMenggunakan Simple Stitcher...")
        images = [cv2.imread(path) for path in image_paths]
        result = simple_stitch(images)
    else:
        print("\nMenggunakan Advanced Stitcher...")
        result = advanced_stitch(image_paths)
    
    if result is not None:
        print("\nStitching berhasil!")
        
        # Crop border hitam
        print("Cropping border...")
        result_cropped = crop_black_borders(result)
        
        # Resize untuk display
        height = result_cropped.shape[0]
        if height > 800:
            result_cropped = imutils.resize(result_cropped, height=800)
        
        # Simpan hasil
        output_path = "output/panorama.jpg"
        os.makedirs("output", exist_ok=True)
        cv2.imwrite(output_path, result_cropped)
        print(f"Hasil disimpan di: {output_path}")
        
        # Tampilkan hasil
        cv2.imshow("Panorama Result", result_cropped)
        print("\nTekan tombol apapun untuk keluar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("\nStitching gagal!")
        print("Tips:")
        print("- Pastikan gambar memiliki overlap yang cukup (30-50%)")
        print("- Pastikan gambar diambil dari sudut yang sama")
        print("- Pastikan pencahayaan konsisten")
