import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def restore_document(image_path, show_steps=True):
    """
    Enhanced document restoration pipeline with improved clarity.
    
    Improvements over basic version:
    - Better noise reduction with bilateral filtering
    - Adaptive thresholding for uneven lighting
    - Larger, optimized kernel sizes
    - Sharpening for crisper text
    - Background normalization
    - More robust morphological operations
    """
    
    # ---------------------------------------------------------
    # Step 1: Read Image
    # ---------------------------------------------------------
    original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if original_img is None:
        print(f"Error: Could not load image at {image_path}")
        return None
    
    print(f"Image loaded: {original_img.shape}")
    
    # ---------------------------------------------------------
    # Step 2: Background Normalization
    # ---------------------------------------------------------
    # Remove uneven illumination and background variations
    # This is crucial for documents with shadows or yellowing
    background = cv2.morphologyEx(original_img, cv2.MORPH_DILATE, 
                                   np.ones((15, 15), np.uint8))
    normalized_img = cv2.divide(original_img, background, scale=255)
    
    # ---------------------------------------------------------
    # Step 3: Advanced Noise Reduction
    # ---------------------------------------------------------
    # Bilateral filter preserves edges while removing noise
    # Better than median filter for text documents
    denoised_img = cv2.bilateralFilter(normalized_img, d=9, 
                                        sigmaColor=75, sigmaSpace=75)
    
    # Additional median filter for salt-and-pepper noise
    denoised_img = cv2.medianBlur(denoised_img, 5)  # Increased from 3 to 5
    
    # ---------------------------------------------------------
    # Step 4: Contrast Enhancement
    # ---------------------------------------------------------
    # CLAHE with optimized parameters
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(denoised_img)
    
    # ---------------------------------------------------------
    # Step 5: Sharpening
    # ---------------------------------------------------------
    # Unsharp masking to make text crisper
    gaussian_blur = cv2.GaussianBlur(enhanced_img, (0, 0), 3.0)
    sharpened_img = cv2.addWeighted(enhanced_img, 1.5, gaussian_blur, -0.5, 0)
    
    # ---------------------------------------------------------
    # Step 6: Adaptive Thresholding
    # ---------------------------------------------------------
    # Better than Otsu's for documents with uneven lighting
    # Using both methods and combining them
    
    # Method 1: Adaptive Gaussian Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        sharpened_img, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        blockSize=15,  # Size of pixel neighborhood
        C=10  # Constant subtracted from mean
    )
    
    # Method 2: Otsu's thresholding (for comparison)
    _, otsu_thresh = cv2.threshold(sharpened_img, 0, 255, 
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Combine both methods (taking the intersection gives best results)
    combined_thresh = cv2.bitwise_and(adaptive_thresh, otsu_thresh)
    
    # ---------------------------------------------------------
    # Step 7: Morphological Operations
    # ---------------------------------------------------------
    # Remove small noise spots
    kernel_opening = np.ones((2, 2), np.uint8)
    cleaned_img = cv2.morphologyEx(combined_thresh, cv2.MORPH_OPEN, kernel_opening)
    
    # Fill gaps in text (closing operation)
    kernel_closing = np.ones((3, 3), np.uint8)  # Increased from 2x2
    final_output = cv2.morphologyEx(cleaned_img, cv2.MORPH_CLOSE, kernel_closing)
    
    # ---------------------------------------------------------
    # Step 8: Final Enhancement (Optional)
    # ---------------------------------------------------------
    # Slight dilation to make text bolder if needed
    # kernel_dilate = np.ones((2, 2), np.uint8)
    # final_output = cv2.dilate(final_output, kernel_dilate, iterations=1)
    
    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    if show_steps:
        titles = [
            '1. Original Image', 
            '2. Normalized Background',
            '3. Denoised (Bilateral + Median)', 
            '4. Enhanced Contrast (CLAHE)',
            '5. Sharpened',
            '6. Adaptive Threshold',
            '7. Otsu Threshold',
            '8. Combined Threshold',
            '9. Final Output (Morphology)'
        ]
        
        images = [
            original_img, 
            normalized_img,
            denoised_img, 
            enhanced_img,
            sharpened_img,
            adaptive_thresh,
            otsu_thresh,
            combined_thresh,
            final_output
        ]
        
        plt.figure(figsize=(20, 12))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)
            plt.title(titles[i], fontsize=11, fontweight='bold')
            plt.axis('off')
        
        plt.tight_layout()
        
        # Save in the same directory as the input image (cross-platform)
        base_path = os.path.splitext(image_path)[0]
        steps_path = f"{base_path}_restoration_steps.png"
        plt.savefig(steps_path, dpi=150, bbox_inches='tight')
        print(f"Restoration steps saved to: {steps_path}")
        
        plt.show()
    
    # ---------------------------------------------------------
    # Save Output
    # ---------------------------------------------------------
    base_path = os.path.splitext(image_path)[0]
    output_path = f"{base_path}_restored.png"
    cv2.imwrite(output_path, final_output)
    print(f"Restored image saved to: {output_path}")
    
    return final_output


def restore_document_simple(image_path):
    """
    Simplified version with just the best settings for quick processing.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return None
    
    # Background normalization
    background = cv2.morphologyEx(img, cv2.MORPH_DILATE, np.ones((15, 15), np.uint8))
    img = cv2.divide(img, background, scale=255)
    
    # Denoise
    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = cv2.medianBlur(img, 5)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    # Sharpen
    blur = cv2.GaussianBlur(img, (0, 0), 3.0)
    img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    
    # Threshold
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, 15, 10)
    
    # Clean up
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    
    base_path = os.path.splitext(image_path)[0]
    output_path = f"{base_path}_restored.png"
    cv2.imwrite(output_path, img)
    print(f"Restored image saved to: {output_path}")
    
    return img


# =========================================================
# Execution
# =========================================================
if __name__ == "__main__":
    # Full pipeline with visualization
    restore_document('sample_document.jpg', show_steps=True)
    
    # Or use simplified version
    # restore_document_simple('sample_document.jpg')