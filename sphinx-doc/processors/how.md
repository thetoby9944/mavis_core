

# ImageProcessor Example


Define some simple settings.
One input column, and a single parameter.
Since mavis is based on steamlit, we use streamlit to congiure the parameter.

The main task is to overwrite `parameter_block`. Do not forget to call `super()`.

```python
import streamlit as st
from mavis.config import DefaultSettings

class MyConfig(DefaultSettings):
    _input_labels = ["Input Images"]
    _output_label = "Output Label"

    divisor: float = 2.

    def parameter_block(self):
        super().parameter_block()
        self.divisor = st.number_input(
            "Divide the pixel values by",
            value=self.divisor,
        )
```

Now define a Processor using the settings from above.
The main task here is to overwrite `process_one`

```python
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from mavis.ui.processors.image import ImageProcessor


class MyImageProcessor(ImageProcessor):
    config = MyConfig()

    def process_one(self, input_1: Union[str, Path]) -> Image.Image:
        img_pil = Image.open(input_1).convert("RGB")
        img_np = np.asarray(img_pil)
        result = img_np / self.config.divisor
        return Image.fromarray(result.astype(np.uint8))

    
# Finally run the processor
MyImageProcessor()
```



You can use the link below to install this template.
Make sure mavis is running on `localhost:8501` 
or adjust the link in the browser 
to point to your mavis host.

[Install locally](http://localhost:8501?package=Classical&module=Image_Processor_Template&code=eJx1k8FqGzEQhu96CrGnFbgq7akY3EMaKIZATZucQli0zqwjqpWENDL123e00tpeh%2BhgmJlP%2F8z%2BGg%2FBjdwrfDO653r0LiDfUciGXMCT1%2FYw55%2BsdpaxGtk0%2BhNXkVs%2FpyIGUKPRmNMRi8Zu%2BzALbEd1AFbSozrqKPfODvrc4B4GlQz%2BAURqG6%2FBpKUPbg8xuhClzkIL1d1cZIztjYqRP8LojUL4MbVob7TFmnE6nbY%2BYWdUDybyDX9utjlRNGPzUiCX8EwR1PyaYv6Q44ZNzKumKV1Y88E4hQR9lbUAA%2Fkb1AgIoeuN2%2F9tI5ihDpBPTB5CK%2BQtJi4EXZC1BWlHlOR%2BT%2BA0fnvm8mnuiXsFjm%2FAvf5HEx%2BVSRB5f2pWC3LKb661L3Xxzselze0yrF9Tn3Nza7648qJc6ZyFyYcVL0%2FwZV3265mWaDWt4Ivgn76Xl5DT78UxPR46r%2FNblLLzYNuqI%2FJSHSFg2%2Fz%2BedeIxSXr6Y71UkUVgjq1VegCBYi0JQRV%2BnPxvnzYbNMVjSnYOkTe1qJaRKgJ%2FX%2BgpXZJW%2FwmsqcfuCnYf0niKAw%3D)




