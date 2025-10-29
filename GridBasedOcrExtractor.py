import os
import cv2
import numpy as np
import subprocess
import csv
import json

class GridBasedOcrExtractor:

    def __init__(self, original_image, grid_result, csv_output_path="extracted_table.csv", json_output_path=None):
        self.original_image = original_image
        self.grid_result = grid_result
        self.cells = grid_result['cells']
        self.table_data = []
        self.csv_output_path = csv_output_path
        self.json_output_path = json_output_path or csv_output_path.replace('.csv', '.json')

    def execute(self):
        self.extract_cell_contents()
        self.generate_csv_file()
        self.generate_json_file()
        return self.table_data

    def extract_cell_contents(self):
        """Extract text from each cell using grid boundaries"""
        for row_idx, row in enumerate(self.cells):
            row_data = []
            for cell in row:
                # Add padding to avoid cutting off text at edges
                padding = 5
                x = max(0, cell['x'] + padding)
                y = max(0, cell['y'] + padding)
                w = max(1, cell['width'] - 2 * padding)
                h = max(1, cell['height'] - 2 * padding)
                
                # Ensure coordinates are within image bounds
                if x + w > self.original_image.shape[1]:
                    w = self.original_image.shape[1] - x
                if y + h > self.original_image.shape[0]:
                    h = self.original_image.shape[0] - y
                
                # Crop the cell from original image
                cell_image = self.original_image[y:y+h, x:x+w]
                
                # Save cell image for debugging
                self.store_cell_image(f"row_{row_idx}_col_{cell['col']}.jpg", cell_image)
                
                # Preprocess cell for better OCR
                processed_cell = self.preprocess_cell_for_ocr(cell_image)
                
                # Save processed cell for debugging
                self.store_cell_image(f"row_{row_idx}_col_{cell['col']}_processed.jpg", processed_cell)
                
                # Perform OCR on the cell
                cell_text = self.ocr_cell(processed_cell, cell['row'], cell['col'])
                
                row_data.append(cell_text)
            
            self.table_data.append(row_data)

    def preprocess_cell_for_ocr(self, cell_image):
        """Preprocess cell image for better OCR results"""
        # Convert to grayscale if not already
        if len(cell_image.shape) == 3:
            gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_image
        
        # Apply thresholding to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
        
        # Resize if image is too small (helps OCR)
        height, width = denoised.shape
        if height < 50 or width < 50:
            scale = max(50 / height, 50 / width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            denoised = cv2.resize(denoised, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return denoised

    def ocr_cell(self, cell_image, row, col):
        """Perform OCR on a single cell"""
        # Save temporary image for tesseract
        temp_path = f"./temp_cells/cell_r{row}_c{col}.jpg"
        os.makedirs("./temp_cells", exist_ok=True)
        cv2.imwrite(temp_path, cell_image)
        
        # Run tesseract with settings optimized for table cells
        # PSM 6: Assume a single uniform block of text
        # This allows multi-line content within a cell
        output = subprocess.getoutput(
            f'tesseract {temp_path} - -l eng --oem 3 --psm 6 --dpi 300'
        )
        
        # Clean up the output
        output = output.strip()
        
        # Remove all '|' characters
        output = output.replace('|', '')
        
        # Replace multiple spaces with single space
        output = ' '.join(output.split())
        
        # Remove newlines but preserve them if they seem intentional
        lines = [line.strip() for line in output.split('\n') if line.strip()]
        output = ' '.join(lines)
        
        # Final trim
        output = output.strip()
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
        
        return output

    def generate_csv_file(self):
        """Generate CSV file from extracted table data"""
        with open(self.csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            
            for row in self.table_data:
                csv_writer.writerow(row)
        
        print(f"  💾 CSV saved: {self.csv_output_path}")

    def generate_json_file(self):
        """Generate JSON file from extracted table data"""
        # Create structured JSON with headers
        if not self.table_data:
            json_data = []
        elif len(self.table_data) == 1:
            # Only header row, return as array
            json_data = self.table_data
        else:
            # First row as headers
            headers = self.table_data[0]
            json_data = []
            
            for row in self.table_data[1:]:
                row_dict = {}
                for idx, header in enumerate(headers):
                    value = row[idx] if idx < len(row) else ""
                    row_dict[header] = value
                json_data.append(row_dict)
        
        # Write to JSON file
        with open(self.json_output_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(json_data, jsonfile, indent=2, ensure_ascii=False)
        
        print(f"  💾 JSON saved: {self.json_output_path}")
        print(f"  📊 Rows: {len(self.table_data)}, Columns: {len(self.table_data[0]) if self.table_data else 0}")

    def store_cell_image(self, file_name, image):
        """Store individual cell images for debugging"""
        path = "./process_images/extracted_cells/"
        os.makedirs(path, exist_ok=True)
        file_path = path + file_name
        cv2.imwrite(file_path, image)

    def print_table_preview(self):
        """Print a preview of the extracted table"""
        print("\n" + "="*80)
        print("TABLE PREVIEW")
        print("="*80)
        
        for row_idx, row in enumerate(self.table_data):
            print(f"\nRow {row_idx}:")
            for col_idx, cell in enumerate(row):
                cell_preview = cell[:50] + "..." if len(cell) > 50 else cell
                print(f"  Col {col_idx}: {cell_preview}")
        
        print("\n" + "="*80)