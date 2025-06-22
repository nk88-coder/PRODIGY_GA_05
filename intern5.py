





import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Load & preprocess image
def load_and_process_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return tf.expand_dims(img, axis=0)

# Deprocess image
def deprocess_img(processed_img):
    x = processed_img[0].numpy()
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]  # BGR to RGB
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Define layers
content_layers = ['block5_conv2']
style_layers = [
    'block1_conv1', 'block2_conv1',
    'block3_conv1', 'block4_conv1',
    'block5_conv1'
]
num_content = len(content_layers)
num_style = len(style_layers)

# VGG19 model
def get_model():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    return tf.keras.Model(vgg.input, model_outputs)

# Gram matrix for style
def gram_matrix(tensor):
    channels = int(tensor.shape[-1])
    a = tf.reshape(tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

# Extract style and content features
def get_feature_representations(model, content_img, style_img):
    content_outputs = model(content_img)
    style_outputs = model(style_img)
    style_features = [gram_matrix(style) for style in style_outputs[:num_style]]
    content_features = [content for content in content_outputs[num_style:]]
    return style_features, content_features

# Loss function
def compute_loss(model, loss_weights, generated_img, gram_style_features, content_features):
    model_outputs = model(generated_img)
    style_output_features = model_outputs[:num_style]
    content_output_features = model_outputs[num_style:]

    style_score = 0
    content_score = 0
    weight_style, weight_content = loss_weights

    for target_style, gen_style in zip(gram_style_features, style_output_features):
        style_score += tf.reduce_mean(tf.square(gram_matrix(gen_style) - target_style))
    for target_content, gen_content in zip(content_features, content_output_features):
        content_score += tf.reduce_mean(tf.square(gen_content - target_content))

    style_score *= weight_style / num_style
    content_score *= weight_content / num_content
    total_loss = style_score + content_score
    return total_loss, style_score, content_score

# Training step
@tf.function()
def train_step(generated_img, model, loss_weights, gram_style_features, content_features, optimizer):
    with tf.GradientTape() as tape:
        total_loss, style_score, content_score = compute_loss(
            model, loss_weights, generated_img, gram_style_features, content_features
        )
    grad = tape.gradient(total_loss, generated_img)
    optimizer.apply_gradients([(grad, generated_img)])
    generated_img.assign(tf.clip_by_value(generated_img, -1.0, 1.0))

# Full pipeline
def run_style_transfer(content_path, style_path, iterations=300, style_weight=1e-2, content_weight=1e4):
    content_img = load_and_process_img(content_path)
    style_img = load_and_process_img(style_path)
    generated_img = tf.Variable(content_img, dtype=tf.float32)

    model = get_model()
    optimizer = tf.optimizers.Adam(learning_rate=5.0)
    style_features, content_features = get_feature_representations(model, content_img, style_img)
    loss_weights = (style_weight, content_weight)

    for i in range(iterations):
        train_step(generated_img, model, loss_weights, style_features, content_features, optimizer)
        if i % 50 == 0:
            print(f"Iteration {i} complete")

    final = deprocess_img(generated_img)
    return final

# === RUNNING SECTION ===
def get_image_paths():
    try:
        # Try Colab file upload
        from google.colab import files
        print("📤 Please upload Content and Style images (in order)...")
        uploaded = files.upload()

        file_names = list(uploaded.keys())
        if len(file_names) < 2:
            raise Exception("Please upload **two images**: one for content, one for style.")

        content_path = file_names[0]
        style_path = file_names[1]

    except Exception:
        print("⚠️ Not in Colab or upload failed. Switching to manual path input.")
        content_path = input("📂 Enter the full path to your Content image: ").strip()
        style_path = input("🎨 Enter the full path to your Style image: ").strip()

        if not os.path.exists(content_path) or not os.path.exists(style_path):
            raise FileNotFoundError("❌ Invalid path provided!")

    return content_path, style_path

# Get image paths
content_path, style_path = get_image_paths()
print(f"\n🖼️ Content: {content_path}\n🎨 Style: {style_path}")

# Run Neural Style Transfer
output = run_style_transfer(content_path, style_path, iterations=300)

# Save and display output
output_img_path = "stylized_output.jpg"
Image.fromarray(output).save(output_img_path)

print("\n✅ Stylized image saved as stylized_output.jpg")
plt.imshow(output)
plt.axis("off")
plt.title("Stylized Output")
plt.show()


