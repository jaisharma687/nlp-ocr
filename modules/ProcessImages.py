import os
import shutil
import modules.TableExtractor as te
import modules.TableGridDetector as tgd
import modules.GridBasedOcrExtractor as gboe
from modules.PdfToImage import convert_pdfs_to_images

def clean_temp_folders():
    """Clean temporary processing folders"""
    temp_folders = ["process_images", "temp_cells"]
    
    for folder in temp_folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

def process_all_images():
    # Clean temporary folders before starting
    clean_temp_folders()
    
    # Setup folders
    input_folder = "Dataset_Images"
    output_csv_folder = "output_csv"
    output_json_folder = "output_json"
    output_images_folder = "images"
    
    # Create output directories
    os.makedirs(output_csv_folder, exist_ok=True)
    os.makedirs(output_json_folder, exist_ok=True)
    os.makedirs(output_images_folder, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    total_images = len(image_files)
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING: {total_images} images found")
    print(f"{'='*80}\n")
    
    successful = 0
    failed = 0
    
    for idx, image_file in enumerate(image_files, 1):
        try:
            print(f"\n[{idx}/{total_images}] Processing: {image_file}")
            print("-" * 80)
            
            # Get image path and base name (without extension)
            image_path = os.path.join(input_folder, image_file)
            base_name = os.path.splitext(image_file)[0]
            
            # Create folder for this image's process images
            image_process_folder = os.path.join(output_images_folder, base_name)
            os.makedirs(image_process_folder, exist_ok=True)
            
            # Extract table from image
            print(f"  📄 Extracting table...")
            table_extractor = te.TableExtractor(image_path)
            perspective_corrected_image = table_extractor.execute()
            
            # Detect grid structure
            print(f"  🔍 Detecting grid structure...")
            grid_detector = tgd.TableGridDetector(perspective_corrected_image)
            grid_result = grid_detector.execute()
            
            # Print grid information
            print(f"  📏 Detected {len(grid_result['vertical_lines'])} vertical lines")
            print(f"  📏 Detected {len(grid_result['horizontal_lines'])} horizontal lines")
            print(f"  📊 Found {len(grid_result['cells'])} rows")
            if grid_result['cells']:
                print(f"  📊 Found {len(grid_result['cells'][0])} columns")
            
            # Extract cell contents using grid-based OCR
            print(f"  📝 Extracting cell contents...")
            ocr_extractor = gboe.GridBasedOcrExtractor(
                perspective_corrected_image, 
                grid_result,
                csv_output_path=os.path.join(output_csv_folder, f"{base_name}.csv"),
                json_output_path=os.path.join(output_json_folder, f"{base_name}.json")
            )
            table_data = ocr_extractor.execute()
            
            # Move process images to the specific image folder
            move_process_images(base_name, image_process_folder)
            
            # Clean temp_cells folder after each image
            if os.path.exists("temp_cells"):
                shutil.rmtree("temp_cells")
                os.makedirs("temp_cells", exist_ok=True)
            
            print(f"  ✅ Successfully processed {image_file}")
            successful += 1
            
        except Exception as e:
            print(f"  ❌ Error processing {image_file}: {str(e)}")
            failed += 1
            continue
    
    # Final cleanup of temporary folders
    print(f"\n🧹 Cleaning up temporary folders...")
    if os.path.exists("process_images"):
        shutil.rmtree("process_images")
    if os.path.exists("temp_cells"):
        shutil.rmtree("temp_cells")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"✅ Successful: {successful}/{total_images}")
    print(f"❌ Failed: {failed}/{total_images}")
    print(f"\n📁 CSV files saved to: {output_csv_folder}/")
    print(f"📁 JSON files saved to: {output_json_folder}/")
    print(f"📁 Process images saved to: {output_images_folder}/")
    print(f"{'='*80}\n")

def move_process_images(base_name, target_folder):
    """Move all process images for a specific file to its folder"""
    source_folders = [
        "process_images/table_extractor",
        "process_images/table_grid_detector",
        "process_images/extracted_cells"
    ]
    
    for source_folder in source_folders:
        if os.path.exists(source_folder):
            # Create corresponding subfolder in target
            folder_name = os.path.basename(source_folder)
            target_subfolder = os.path.join(target_folder, folder_name)
            os.makedirs(target_subfolder, exist_ok=True)
            
            # Move all files from source to target
            for file in os.listdir(source_folder):
                source_file = os.path.join(source_folder, file)
                target_file = os.path.join(target_subfolder, file)
                if os.path.isfile(source_file):
                    shutil.move(source_file, target_file)
            
            # Remove empty source folder
            try:
                os.rmdir(source_folder)
            except:
                pass

def convert_and_process_all_images():
    convert_pdfs_to_images()
    process_all_images()

if __name__ == "__main__":
    convert_and_process_all_images()