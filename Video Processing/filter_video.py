"""
Video Filter - Black & White Oscilloscope Effect
Processes pre-recorded MP4 videos with the music video filter effect.

Usage:
    python filter_video.py input.mp4
    python filter_video.py input.mp4 -o output.mp4
    python filter_video.py input.mp4 --preview --quality high
"""

import cv2
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm
from rembg import new_session, remove

class VideoFilter:
    def __init__(self, quality='medium', show_preview=False):
        """Initialize the video filter with quality settings."""
        self.session = new_session("u2net", providers=["CPUExecutionProvider"])

        # Quality presets
        self.quality_settings = {
            'low': {
                'edge_threshold1': 30,
                'edge_threshold2': 80,
                'mask_blur': 7,
                'bloom_intensity': 0.3,
                'bloom_size': 11
            },
            'medium': {
                'edge_threshold1': 20,
                'edge_threshold2': 60,
                'mask_blur': 11,
                'bloom_intensity': 0.5,
                'bloom_size': 15
            },
            'high': {
                'edge_threshold1': 15,
                'edge_threshold2': 50,
                'mask_blur': 11,
                'bloom_intensity': 0.6,
                'bloom_size': 21
            },
            'ultra': {
                'edge_threshold1': 10,
                'edge_threshold2': 40,
                'mask_blur': 9,
                'bloom_intensity': 0.7,
                'bloom_size': 25
            }
        }

        self.settings = self.quality_settings.get(quality, self.quality_settings['medium'])
        self.show_preview = show_preview

        print(f"--- Quality: {quality.upper()} ---")
        print(f"--- Edge detection: {self.settings['edge_threshold1']}-{self.settings['edge_threshold2']} ---")
        print(f"--- Bloom intensity: {self.settings['bloom_intensity']} ---")

    def process_frame(self, frame):
        """Process a single frame with the oscilloscope effect."""
        # Remove background to get mask
        result = remove(frame, session=self.session)

        if result.shape[2] == 4:
            mask = result[:, :, 3] / 255.0
            mask_blurred = cv2.GaussianBlur(mask, (self.settings['mask_blur'], self.settings['mask_blur']), 0)
            condition = np.stack((mask_blurred,) * 3, axis=-1) > 0.2

            # Edge detection (oscilloscope effect)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(gray_blurred, self.settings['edge_threshold1'], self.settings['edge_threshold2'])

            person_lines = np.zeros_like(frame)
            person_lines[edges > 0] = [255, 255, 255]

            # Black background with scanlines
            bg = np.zeros_like(frame)
            for i in range(0, frame.shape[0], 10):
                cv2.line(bg, (0, i), (frame.shape[1], i), (30, 30, 30), 1)

            # Combine and add bloom
            output = np.where(condition, person_lines, bg)
            bloom = cv2.GaussianBlur(output, (self.settings['bloom_size'], self.settings['bloom_size']), 0)
            final_frame = cv2.addWeighted(output, 1.0, bloom, self.settings['bloom_intensity'], 0)

            return final_frame

        return frame

    def process_video(self, input_path, output_path=None):
        """Process the entire video file."""
        if not os.path.exists(input_path):
            print(f"Error: Input file '{input_path}' not found.")
            return False

        # Generate output filename if not provided
        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_filtered{ext}"

        print(f"\n--- VIDEO FILTER ---")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")

        # Open video file
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return False

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total frames: {frame_count}")
        print(f"\nProcessing...")

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Process frames with progress bar
        try:
            for frame_idx in tqdm(range(frame_count), desc="Filtering", unit="frames"):
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                filtered_frame = self.process_frame(frame)
                out.write(filtered_frame)

                # Show preview if enabled
                if self.show_preview:
                    cv2.imshow('Preview (Press Q to skip preview)', filtered_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.show_preview = False
                        cv2.destroyAllWindows()

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user.")
            cap.release()
            out.release()
            if os.path.exists(output_path):
                os.remove(output_path)
            return False

        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"\n✅ Complete! Output saved to: {output_path}")
        return True

def batch_process(input_files, quality='medium', show_preview=False):
    """Process multiple video files."""
    filter_processor = VideoFilter(quality=quality, show_preview=show_preview)

    results = []
    for input_path in input_files:
        print(f"\n{'='*60}")
        success = filter_processor.process_video(input_path)
        results.append((input_path, success))

    # Summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    for input_path, success in results:
        status = "✅ Success" if success else "❌ Failed"
        print(f"{status}: {input_path}")

    successful = sum(1 for _, s in results if s)
    print(f"\nCompleted: {successful}/{len(input_files)} videos")

def main():
    parser = argparse.ArgumentParser(
        description='Apply black & white oscilloscope effect to video files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.mp4
  %(prog)s video.mp4 -o output.mp4
  %(prog)s video.mp4 --quality high
  %(prog)s video.mp4 --preview
  %(prog)s video1.mp4 video2.mp4 video3.mp4
        """
    )

    parser.add_argument('input', nargs='+', help='Input video file(s) (MP4, AVI, MOV, etc.)')
    parser.add_argument('-o', '--output', help='Output video file path (for single input only)')
    parser.add_argument('-q', '--quality', choices=['low', 'medium', 'high', 'ultra'],
                        default='medium', help='Processing quality (default: medium)')
    parser.add_argument('-p', '--preview', action='store_true',
                        help='Show live preview during processing')
    parser.add_argument('--batch', action='store_true',
                        help='Batch mode for multiple files')

    args = parser.parse_args()

    # Handle single file vs multiple files
    if len(args.input) == 1:
        filter_processor = VideoFilter(quality=args.quality, show_preview=args.preview)
        filter_processor.process_video(args.input[0], args.output)
    else:
        if args.output:
            print("Warning: --output ignored when processing multiple files")
        batch_process(args.input, quality=args.quality, show_preview=args.preview)

if __name__ == "__main__":
    print("=" * 60)
    print(" VIDEO FILTER - Black & White Oscilloscope Effect")
    print("=" * 60)

    print("\nLoading background removal model...")
    print("(This may take a moment on first run)\n")

    main()

    print("\n" + "=" * 60)
    print(" DONE!")
    print("=" * 60)
