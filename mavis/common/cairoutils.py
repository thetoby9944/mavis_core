import cairo
import numpy as np
from PIL import Image


def surface_to_pil(s):
    cario_format = s.get_format()
    if cario_format == cairo.FORMAT_ARGB32:
        pil_mode = 'RGB'
        # Cairo has ARGB. Convert this to RGB for PIL which supports only RGB or
        # RGBA.
        argb_array = np.fromstring(bytes(s.get_data()), 'c').reshape(-1, 4)
        rgb_array = argb_array[:, 2::-1]
        pil_data = rgb_array.reshape(-1).tostring()
    else:
        raise ValueError('Unsupported cairo format: %d' % cario_format)
    pil_image = Image.frombuffer(pil_mode,
                                 (s.get_width(),
                                  s.get_height()),
                                 pil_data, "raw", pil_mode, 0, 1)
    return pil_image.convert('RGB')