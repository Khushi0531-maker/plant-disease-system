"""
Utility functions for Plant Disease Detection:
- Grad-CAM (Explainable AI Attention Heatmap)
- Image Preprocessing and Enhancement
- Agronomic Diagnostic Report Generator
Supports both OpenCV and Matplotlib/Pillow fallback.
"""

import numpy as np
from PIL import Image, ImageFilter
import matplotlib.cm as cm

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def preprocess_image(image_input, target_size=(224, 224)):
    """
    Converts various input formats (PIL, NumPy array, file path)
    into standard normalized tensor format (1, 224, 224, 3) and display PIL image.
    """
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 3 and image_input.shape[2] == 4:
            pil_img = Image.fromarray(image_input).convert("RGB")
        else:
            pil_img = Image.fromarray(image_input)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        pil_img = Image.open(image_input).convert("RGB")

    # Resize with high quality bicubic resampling
    resized_img = pil_img.resize(target_size, Image.Resampling.BILINEAR)
    img_array = np.array(resized_img, dtype=np.float32)
    
    # Model expects (1, H, W, 3) in float32 [0..255]
    tensor = np.expand_dims(img_array, axis=0)
    return tensor, pil_img


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Generates a Grad-CAM heatmap showing the spatial regions that
    influenced the CNN's classification decision.
    """
    try:
        import tensorflow as tf
    except ImportError:
        return None

    try:
        # Check if MobileNetV2 transfer model
        if any("mobilenetv2" in l.name.lower() for l in model.layers):
            base_layer = next(l for l in model.layers if "mobilenetv2" in l.name.lower())
            aug_layer = model.get_layer('data_augmentation') if 'data_augmentation' in [l.name for l in model.layers] else None
            
            with tf.GradientTape() as tape:
                x = tf.cast(img_array, tf.float32)
                if aug_layer:
                    x = aug_layer(x)
                conv_outputs = base_layer(x)
                tape.watch(conv_outputs)
                
                # Forward through subsequent classification head layers
                y = conv_outputs
                started = False
                for layer in model.layers:
                    if started:
                        y = layer(y, training=False)
                    if layer == base_layer:
                        started = True
                
                preds = y
                if pred_index is None:
                    pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]

            grads = tape.gradient(class_channel, conv_outputs)
        else:
            # Custom CNN or sequential model
            conv_layer_idx = None
            for idx in reversed(range(len(model.layers))):
                if isinstance(model.layers[idx], tf.keras.layers.Conv2D):
                    conv_layer_idx = idx
                    break

            if conv_layer_idx is None:
                return None

            with tf.GradientTape() as tape:
                x = tf.cast(img_array, tf.float32)
                for i in range(conv_layer_idx + 1):
                    x = model.layers[i](x, training=False)
                conv_outputs = x
                tape.watch(conv_outputs)

                y = conv_outputs
                for i in range(conv_layer_idx + 1, len(model.layers)):
                    y = model.layers[i](y, training=False)

                preds = y
                if pred_index is None:
                    pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]

            grads = tape.gradient(class_channel, conv_outputs)

        if grads is None:
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()

    except Exception as e:
        print(f"GradCAM notice: {e}")
        return None


def overlay_gradcam(original_pil, heatmap, alpha=0.45):
    """
    Overlays colorized Grad-CAM heatmap onto the original leaf image.
    Works seamlessly with OpenCV or Matplotlib/Pillow.
    """
    orig_np = np.array(original_pil)
    h, w = orig_np.shape[:2]

    if heatmap is None:
        return orig_np

    if HAS_CV2:
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        color_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
        superimposed = cv2.addWeighted(orig_np, 1.0 - alpha, color_map, alpha, 0)
        return superimposed
    else:
        # High quality pure PIL + Matplotlib Jet colormap
        heatmap_pil = Image.fromarray(np.uint8(255 * heatmap)).resize((w, h), Image.Resampling.BILINEAR)
        heatmap_norm = np.array(heatmap_pil, dtype=np.float32) / 255.0
        
        # Apply Jet colormap
        try:
            import matplotlib
            jet = matplotlib.colormaps['jet']
        except Exception:
            import matplotlib.cm as cm
            jet = getattr(cm, 'jet', cm.get_cmap('jet'))
        jet_colors = jet(heatmap_norm)[:, :, :3] # RGB channels
        color_map = np.uint8(255 * jet_colors)

        # Blend
        superimposed = np.uint8((1.0 - alpha) * orig_np + alpha * color_map)
        return superimposed


def generate_printable_report(info_dict: dict, confidence: float, top_classes: list) -> str:
    """
    Generates a clean HTML report card suitable for printing or saving.
    """
    top_table_rows = "".join([
        f"<tr><td><b>{idx+1}. {c['crop']} - {c['disease']}</b></td><td>{c['prob']:.1f}%</td><td>{c['status']}</td></tr>"
        for idx, c in enumerate(top_classes)
    ])

    html = f"""
    <div style="font-family: Arial, sans-serif; padding: 25px; border: 2px solid #2e7d32; border-radius: 12px; background: #fafafa; color: #1e293b;">
        <div style="display: flex; justify-content: space-between; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; margin-bottom: 16px;">
            <div>
                <h2 style="color: #2e7d32; margin: 0;">🌿 FloraScan AI - Diagnostic Report</h2>
                <p style="margin: 4px 0; color: #64748b; font-size: 13px;">Automated Deep Learning Plant Pathology Analysis</p>
            </div>
            <div style="text-align: right;">
                <span style="background: {'#16a34a' if info_dict['status'] == 'Healthy' else '#dc2626'}; color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 14px;">
                    {info_dict['status'].upper()}
                </span>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;">
            <div style="background: white; padding: 12px 16px; border-radius: 8px; border: 1px solid #e2e8f0;">
                <p style="margin: 0; font-size: 12px; color: #64748b;">Primary Diagnosis</p>
                <h3 style="margin: 4px 0; color: #0f172a;">{info_dict['crop']} - {info_dict['disease']}</h3>
                <p style="margin: 0; font-size: 13px; font-style: italic; color: #475569;">Pathogen: {info_dict['scientific_name']}</p>
            </div>
            <div style="background: white; padding: 12px 16px; border-radius: 8px; border: 1px solid #e2e8f0;">
                <p style="margin: 0; font-size: 12px; color: #64748b;">Diagnostic Confidence</p>
                <h3 style="margin: 4px 0; color: #16a34a;">{confidence:.2f}%</h3>
                <p style="margin: 0; font-size: 13px; color: #475569;">Pathogen Type: <b>{info_dict['pathogen_type']}</b></p>
            </div>
        </div>

        <div style="margin-bottom: 16px;">
            <h4 style="margin: 0 0 6px 0; color: #0f172a;">🔍 Clinical Symptoms</h4>
            <p style="margin: 0; font-size: 14px; color: #334155; line-height: 1.5;">{info_dict['symptoms']}</p>
        </div>

        <div style="background: #ecfdf5; border-left: 4px solid #10b981; padding: 12px; margin-bottom: 12px; border-radius: 4px;">
            <h4 style="margin: 0 0 4px 0; color: #065f46;">🌱 Recommended Organic & Biological Remedy</h4>
            <p style="margin: 0; font-size: 13px; color: #047857;">{info_dict['organic_treatment']}</p>
        </div>

        <div style="background: #fef2f2; border-left: 4px solid #ef4444; padding: 12px; margin-bottom: 12px; border-radius: 4px;">
            <h4 style="margin: 0 0 4px 0; color: #991b1b;">🧪 Chemical Intervention & Active Ingredients</h4>
            <p style="margin: 0; font-size: 13px; color: #b91c1c;">{info_dict['chemical_treatment']}</p>
        </div>

        <div style="background: #f0f9ff; border-left: 4px solid #0284c7; padding: 12px; margin-bottom: 20px; border-radius: 4px;">
            <h4 style="margin: 0 0 4px 0; color: #075985;">🛡️ Prevention & Cultural Practices</h4>
            <p style="margin: 0; font-size: 13px; color: #0369a1;">{info_dict['prevention']} | <i>Irrigation: {info_dict['irrigation_tip']}</i></p>
        </div>

        <div>
            <h4 style="margin: 0 0 8px 0; color: #0f172a;">📊 Top Multiclass Probabilities</h4>
            <table style="width: 100%; border-collapse: collapse; font-size: 13px; text-align: left;">
                <tr style="background: #f1f5f9; border-bottom: 2px solid #cbd5e1;">
                    <th style="padding: 6px 10px;">Predicted Class</th>
                    <th style="padding: 6px 10px;">Confidence</th>
                    <th style="padding: 6px 10px;">Status</th>
                </tr>
                {top_table_rows}
            </table>
        </div>
        <p style="text-align: center; margin-top: 20px; font-size: 11px; color: #94a3b8;">Generated by FloraScan Deep Learning Vision System</p>
    </div>
    """
    return html
