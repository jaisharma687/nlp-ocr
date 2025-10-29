import os
from pdf2image import convert_from_path

def convert_pdfs_to_images():
    """Convert all PDFs in Dataset folder to images in Dataset_Images folder"""
    input_folder = "Dataset"
    output_folder = "Dataset_Images"
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all PDF files
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print("📝 No PDF files found in Dataset folder")
        return 0
    
    print(f"\n{'='*80}")
    print(f"PDF CONVERSION: {len(pdf_files)} PDFs found")
    print(f"{'='*80}\n")
    
    converted = 0
    for pdf_file in pdf_files:
        try:
            pdf_path = os.path.join(input_folder, pdf_file)
            
            # Convert PDF to image (first page only)
            print(f"📄 Converting: {pdf_file}")
            pages = convert_from_path(pdf_path, dpi=300)
            image = pages[0]
            
            # Save image with same name but .png extension
            image_name = os.path.splitext(pdf_file)[0] + ".png"
            image_path = os.path.join(output_folder, image_name)
            image.save(image_path, "PNG")
            
            print(f"  ✅ Saved: {image_name}")
            converted += 1
            
        except Exception as e:
            print(f"  ❌ Error converting {pdf_file}: {str(e)}")
    
    print(f"\n{'='*80}")
    print(f"✅ Converted {converted}/{len(pdf_files)} PDFs successfully")
    print(f"{'='*80}\n")
    
    return converted

if __name__ == "__main__":
    convert_pdfs_to_images()