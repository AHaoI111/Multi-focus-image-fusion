import cv2
import numpy as np
import concurrent.futures


def merge_images(images):
    # images为img列表
    def select_values_from_channels(channels):
        height, width, c = channels.shape
        magnitudes = np.zeros((height, width, c), dtype=np.float32)

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(process_channel, channels[:, :, i]) for i in range(c)]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                result = future.result()
                magnitudes[:, :, i] = result

        max_channel_index = np.argmax(magnitudes, axis=2)
        selected_pixels = channels[np.arange(height)[:, None], np.arange(width), max_channel_index]
        return selected_pixels

    def process_channel(channel):
        sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(sobel_x, sobel_y)
        ksize = (21, 21)
        smoothed_image = cv2.blur(magnitude, ksize)
        return smoothed_image

    def focus_images(imgs):
        img_array = np.array(imgs)
        return img_array[:, :, :, 0], img_array[:, :, :, 1], img_array[:, :, :, 2]

    img_b, img_g, img_r = focus_images(images)

    channels_b = np.dstack(img_b) if img_b is not None else None
    channels_g = np.dstack(img_g) if img_g is not None else None
    channels_r = np.dstack(img_r) if img_r is not None else None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(select_values_from_channels, [channels_b, channels_g, channels_r]))

    selected_pixel_b, selected_pixel_g, selected_pixel_r = results

    merged_image = cv2.merge([selected_pixel_b, selected_pixel_g, selected_pixel_r])

    return merged_image


