import cv2
import numpy as np
import imutils

class ImageStitcher:
    def __init__(self):
        self.isv3 = imutils.is_cv3(or_better=True)
        
    def stitch(self, images, ratio=0.75, reprojThresh=4.0):
        """Menggabungkan beberapa gambar menjadi panorama (menggunakan homography)"""
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)
        
        # Match fitur
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
        
        if M is None:
            return None
        
        (matches, H, status) = M
        
        # Hitung bounding box yang tepat
        hA, wA = imageA.shape[:2]
        hB, wB = imageB.shape[:2]
        cornersA = np.float32([[0,0], [0,hA], [wA,hA], [wA,0]]).reshape(-1,1,2)
        warped_cornersA = cv2.perspectiveTransform(cornersA, H)
        cornersB = np.float32([[0,0], [0,hB], [wB,hB], [wB,0]]).reshape(-1,1,2)
        all_corners = np.concatenate((warped_cornersA, cornersB), axis=0)
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        trans = [-xmin, -ymin]
        
        # Canvas yang pas
        out_w = xmax - xmin
        out_h = ymax - ymin
        H_translate = np.array([[1, 0, trans[0]], [0, 1, trans[1]], [0, 0, 1]])
        
        result = cv2.warpPerspective(imageA, H_translate @ H, (out_w, out_h))
        
        # Blending dengan distance transform untuk transisi halus
        y_off = trans[1]
        x_off = trans[0]
        y_end = min(y_off + hB, out_h)
        x_end = min(x_off + wB, out_w)
        
        region = result[y_off:y_end, x_off:x_end]
        imageB_crop = imageB[:y_end-y_off, :x_end-x_off]
        
        # Buat mask untuk area yang terisi di region dan imageB_crop
        mask_A = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) > 0
        mask_B = cv2.cvtColor(imageB_crop, cv2.COLOR_BGR2GRAY) > 0
        overlap = mask_A & mask_B
        
        if np.any(overlap):
            # Distance transform untuk smooth blending
            dist_A = cv2.distanceTransform(mask_A.astype(np.uint8), cv2.DIST_L2, 5)
            dist_B = cv2.distanceTransform(mask_B.astype(np.uint8), cv2.DIST_L2, 5)
            
            # Normalisasi weight
            weight_A = dist_A / (dist_A + dist_B + 1e-10)
            weight_B = dist_B / (dist_A + dist_B + 1e-10)
            
            # Blending
            blended = (region.astype(float) * weight_A[:,:,None] + 
                      imageB_crop.astype(float) * weight_B[:,:,None])
            
            result[y_off:y_end, x_off:x_end] = blended.astype(np.uint8)
        else:
            # Tidak ada overlap, langsung copy
            result[y_off:y_end, x_off:x_end] = imageB_crop
        
        return result
    
    def detectAndDescribe(self, image):
        """Deteksi keypoints dan ekstrak fitur"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if self.isv3:
            descriptor = cv2.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(gray, None)
        else:
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)
        
        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)
    
    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        """Match keypoints antara dua gambar"""
        matcher = cv2.BFMatcher()
        rawMatches = matcher.knnMatch(featuresA, featuresB, k=2)
        matches = []
        
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        
        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
            
            return (matches, H, status)
        
        return None
    
    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        """Visualisasi matching"""
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB
        
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        
        return vis

if __name__ == "__main__":
    import sys
    # Ganti path ke folder images
    img_files = ["contoh4/1.jpeg", "contoh1/2.jpeg", "contoh1/3.jpeg"]
    imgs = [cv2.imread(f) for f in img_files]
    if any(img is None for img in imgs):
        print("❌ Salah satu gambar tidak ditemukan.")
        sys.exit(1)

    stitcher = ImageStitcher()
    # Gabungkan gambar satu per satu
    result = imgs[0]
    for next_img in imgs[1:]:
        stitched = stitcher.stitch([result, next_img])
        if stitched is None:
            print("❌ Gagal menggabungkan gambar.")
            sys.exit(1)
        result = stitched

    cv2.imwrite("panorama.png", result)
    print("✅ Panorama disimpan ke panorama.png")
    cv2.imshow("Panorama", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
