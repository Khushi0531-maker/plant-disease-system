"""
Generates botanically realistic synthetic sample leaf images for instant testing.
Creates leaf shapes, primary and secondary venation, and authentic pathological lesions.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def create_leaf_base(width=400, height=400, leaf_color=(45, 135, 55)):
    """Creates a realistic leaf base with organic gradient and venation."""
    img = Image.new("RGB", (width, height), (245, 247, 248))
    draw = ImageDraw.Draw(img)

    # Elliptical / lanceolate leaf mask
    cx, cy = width // 2, height // 2
    
    # Draw leaf body using polygon / bezier
    points = []
    for angle in np.linspace(0, 2 * np.pi, 200):
        # Organic leaf profile equation
        r = 150 * (1.0 - 0.5 * np.cos(angle)) * (0.8 + 0.2 * np.sin(angle * 2))
        x = cx + r * np.cos(angle) * 0.7
        y = cy + r * np.sin(angle) * 1.1
        points.append((x, y))

    draw.polygon(points, fill=leaf_color)

    # Draw central main vein
    draw.line([(cx, cy - 140), (cx, cy + 150)], fill=(75, 175, 80), width=5)
    
    # Draw lateral side veins
    for y_offset in range(-100, 120, 25):
        y_start = cy + y_offset
        draw.line([(cx, y_start), (cx - 70, y_start - 35)], fill=(65, 160, 70), width=2)
        draw.line([(cx, y_start), (cx + 70, y_start - 35)], fill=(65, 160, 70), width=2)

    return img


def add_early_blight_lesions(img):
    """Draws target-board concentric dark brown rings with yellow chlorotic halos."""
    draw = ImageDraw.Draw(img)
    spots = [(170, 180, 28), (230, 220, 35), (200, 270, 22), (150, 250, 18)]
    
    for x, y, r in spots:
        # Yellow chlorotic halo
        draw.ellipse([x - r - 8, y - r - 8, x + r + 8, y + r + 8], fill=(210, 195, 45))
        # Concentric brown rings
        for ring_r in range(r, 0, -4):
            shade = 45 if ring_r % 8 == 0 else 75
            draw.ellipse([x - ring_r, y - ring_r, x + ring_r, y + ring_r], fill=(shade + 20, shade, 20))

    return img.filter(ImageFilter.GaussianBlur(radius=1.2))


def add_late_blight_lesions(img):
    """Draws dark, water-soaked necrotic irregular lesions with pale margins."""
    draw = ImageDraw.Draw(img)
    spots = [(180, 160, 45), (240, 250, 55), (160, 260, 35)]
    
    for x, y, r in spots:
        draw.ellipse([x - r - 5, y - r - 5, x + r + 5, y + r + 5], fill=(130, 140, 65))
        # Water-soaked dark rotting core
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(35, 30, 25))
        draw.ellipse([x - r + 8, y - r + 8, x + r - 8, y + r - 8], fill=(25, 20, 18))

    return img.filter(ImageFilter.GaussianBlur(radius=1.5))


def add_common_rust_pustules(img):
    """Draws scattered raised brick-red/cinnamon brown powdery pustules."""
    draw = ImageDraw.Draw(img)
    np.random.seed(42)
    
    for _ in range(65):
        rx = np.random.randint(140, 260)
        ry = np.random.randint(100, 310)
        size = np.random.randint(4, 9)
        color = (np.random.randint(180, 215), np.random.randint(60, 85), 15)
        draw.ellipse([rx, ry, rx + size, ry + size], fill=color)

    return img.filter(ImageFilter.GaussianBlur(radius=0.8))


def add_apple_scab_lesions(img):
    """Draws olive-green to blackish velvety crusty patches."""
    draw = ImageDraw.Draw(img)
    scabs = [(175, 170, 25), (225, 190, 30), (190, 240, 22), (220, 270, 28)]
    
    for x, y, r in scabs:
        draw.ellipse([x - r - 4, y - r - 4, x + r + 4, y + r + 4], fill=(70, 85, 35))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(40, 45, 28))
        draw.ellipse([x - r + 5, y - r + 5, x + r - 5, y + r - 5], fill=(30, 32, 20))

    return img.filter(ImageFilter.GaussianBlur(radius=1.0))


def generate_all_samples(output_dir="sample_images"):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating Tomato Early Blight...")
    img = create_leaf_base()
    img = add_early_blight_lesions(img)
    img.save(os.path.join(output_dir, "tomato_early_blight.jpg"), quality=95)

    print("Generating Potato Late Blight...")
    img = create_leaf_base(leaf_color=(40, 125, 50))
    img = add_late_blight_lesions(img)
    img.save(os.path.join(output_dir, "potato_late_blight.jpg"), quality=95)

    print("Generating Corn Common Rust...")
    img = create_leaf_base(leaf_color=(75, 150, 45))
    img = add_common_rust_pustules(img)
    img.save(os.path.join(output_dir, "corn_common_rust.jpg"), quality=95)

    print("Generating Apple Scab...")
    img = create_leaf_base(leaf_color=(50, 140, 60))
    img = add_apple_scab_lesions(img)
    img.save(os.path.join(output_dir, "apple_scab.jpg"), quality=95)

    print("Generating Tomato Healthy...")
    img = create_leaf_base(leaf_color=(38, 145, 52))
    img.save(os.path.join(output_dir, "tomato_healthy.jpg"), quality=95)

    print("All sample leaf images generated successfully!")

if __name__ == "__main__":
    generate_all_samples()
