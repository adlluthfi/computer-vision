import cv2
import os

def quick_stitch():
    """Quick stitch menggunakan OpenCV Stitcher"""
    image_folder = "images/"
    images = []
    
    # Load gambar
    for i in range(1, 4):
        img_path = os.path.join(image_folder, f"gambar{i}.png")
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Tidak dapat membaca: {img_path}")
            return
        
        images.append(img)
        print(f"Loaded: gambar{i}.png")
    
    print("\nProses stitching...")
    
    # Buat stitcher
    stitcher = cv2.Stitcher_create()
    
    # Stitch
    status, panorama = stitcher.stitch(images)
    
    if status == cv2.Stitcher_OK:
        print("Stitching berhasil!")
        
        # Simpan hasil
        os.makedirs("output", exist_ok=True)
        cv2.imwrite("output/panorama.jpg", panorama)
        print("Hasil disimpan: output/panorama.jpg")
        
        # Tampilkan
        cv2.imshow("Panorama", panorama)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Stitching gagal! Status: {status}")
        if status == cv2.Stitcher_ERR_NEED_MORE_IMGS:
            print("Error: Butuh lebih banyak gambar")
        elif status == cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:
            print("Error: Gagal estimasi homography")
            print("Pastikan gambar memiliki overlap yang cukup!")
        elif status == cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:
            print("Error: Gagal adjust parameter kamera")

if __name__ == "__main__":
    quick_stitch()
