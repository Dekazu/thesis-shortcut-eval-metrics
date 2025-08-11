import math
from datetime import datetime
from datetime import timedelta
import cv2

import numpy as np
import torch
import torchvision.transforms as T
from PIL import ImageDraw, Image, ImageFilter


def get_artifact_kwargs(config):
    artifact_kwargs = {}
    artifact_type = config.get("artifact_type", None)
    if artifact_type == "channel":
        artifact_kwargs = {
            'op_type': config.get('op_type', 'add'),
            'channel': config.get('channel', 0),
            'value': config.get('value', 100)
        }
    elif artifact_type == "lsb":
        artifact_kwargs = {
            'lsb_trigger': config.get('lsb_trigger', "ThisIsASecretCleverHansTrigger"),
            'lsb_factor': config.get("lsb_factor", 3),
            'start_bit': config.get("start_bit", 0)
        }
    elif artifact_type == "random_mnist":
        artifact_kwargs = {
            'shift_factor': config['shift_factor'],
            'datapath_mnist': config['datapath_mnist']
        }
    elif artifact_type == "ch_time":
        artifact_kwargs = {
            "time_format": config.get("time_format", "time")
        }
    elif artifact_type == "color_mnist":
        artifact_kwargs = {
            "color_id": config["attacked_classes"][0]
        }
    elif artifact_type == "white_color":
        artifact_kwargs = {
            "alpha": config.get("alpha", .3)
        }
    elif artifact_type == "defective_lead":
        artifact_kwargs = {
            "lead_ids": config.get("lead_ids", [1]),
            "seq_length": config.get("seq_length", 100)
        }
    elif artifact_type == "amplified_beat":
        artifact_kwargs = {
            "lead_ids": config.get("lead_ids", [3]),
            "peak_ids": config.get("peak_ids", [2]),
            "amplify_factor": config.get("amplify_factor", 3)
        }
    return artifact_kwargs


def insert_artifact(img, artifact_type, **kwargs):
    if artifact_type == "ch_time":
        return insert_artifact_ch_time(img, **kwargs)
    elif artifact_type == "ch_text":
        return insert_artifact_ch_text(img, **kwargs)
    elif artifact_type == "microscope":
        return insert_microscope(img, **kwargs)
    elif artifact_type == "channel":
        return insert_artifact_channel(img, **kwargs)
    elif artifact_type == "white_color":
        return insert_artifact_white_color(img, **kwargs)
    elif artifact_type == "red_color":
        return insert_artifact_red_color(img, **kwargs)
    elif artifact_type == "lsb":
        return insert_lsb_trigger(img, **kwargs)
    elif artifact_type == "constant_box":
        return insert_constant_box(img, **kwargs)
    elif artifact_type == "random_box":
        return insert_random_box(img, **kwargs)
    elif artifact_type == "random_mnist":
        return insert_random_mnist(img, **kwargs)
    elif artifact_type == "color_mnist":
        return color_digit(img, **kwargs)
    elif artifact_type == "reflection":
        return insert_reflection(img, **kwargs)
    elif artifact_type == "bone":
        return insert_bone(img, **kwargs)
    else:
        raise ValueError(f"Unknown artifact_type: {artifact_type}")

def insert_bone(img, **kwargs):
    """
    Insert a shaded, outlined cartoon bone into `img`.

    Required
    --------
    img : PIL.Image | ndarray

    Keyword options (all optional)
    ------------------------------
    min_size          int   tip-to-tip length lower bound        (20)
    max_size          int   tip-to-tip length upper bound        (60)
    rotate            bool  random 0–360° rotation               (True)
    thickness_ratio   float shaft thickness / length             (0.25)
    knob_ratio        float lobe radius / (shaft/2)              (1.1)
    lobe_gap_ratio    float gap between the two lobes / radius   (0.6)
    outline_thickness int   black outline thickness in px        (2)
    seed              int   RNG seed

    Returns
    -------
    aug_img : PIL.Image
    mask    : torch.Tensor (H, W)  – 1 ⇔ bone interior (incl. outline)
    """

    # ─── parameters ─────────────────────────────────────────────────────────
    min_size   = kwargs.get("min_size", 40)
    max_size   = kwargs.get("max_size", 60)
    rotate     = kwargs.get("rotate", True)
    thick_k    = kwargs.get("thickness_ratio", 0.25)
    knob_k     = kwargs.get("knob_ratio", 1.1)
    gap_k      = kwargs.get("lobe_gap_ratio", 0.6)
    stroke_px  = kwargs.get("outline_thickness", 2)
    seed       = kwargs.get("seed", None)
    if seed is not None:
        np.random.seed(seed)

    # ─── choose bone dimensions ─────────────────────────────────────────────
    base = np.asarray(img).copy()
    H, W = base.shape[:2]
    L    = np.random.randint(min_size, max_size + 1)
    shaft = max(4, int(L * thick_k))
    knob_r = max(3, int(shaft * knob_k / 2))
    gap    = int(knob_r * gap_k)

    # enough padding that rotation never clips anything
    pad  = knob_r + stroke_px + abs(gap) + 2
    canv = np.zeros((L + 2 * pad, L + 2 * pad), np.uint8)        # 1-ch mask

    cx1 = pad + knob_r
    cx2 = pad + L - knob_r - 1
    cy  = pad + L // 2
    shaft_top = cy - shaft // 2
    shaft_bot = cy + shaft // 2

    # ─── draw bone mask (4 lobes + shaft) ───────────────────────────────────
    for x in (cx1, cx2):
        for y in (cy - gap, cy + gap):
            cv2.circle(canv, (x, y), knob_r, 255, -1)
    cv2.rectangle(canv, (cx1, shaft_top), (cx2, shaft_bot), 255, -1)

    # ─── outline (stroke) mask  – dilate minus original ─────────────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * stroke_px + 1, 2 * stroke_px + 1))
    outline = cv2.dilate(canv, kernel) - canv

    # ─── optional rotation ──────────────────────────────────────────────────
    if rotate:
        angle = np.random.uniform(0, 360)
        R     = cv2.getRotationMatrix2D(((L + 2*pad)/2, (L + 2*pad)/2), angle, 1.0)
        canv     = cv2.warpAffine(canv,     R, canv.shape[::-1], flags=cv2.INTER_NEAREST)
        outline  = cv2.warpAffine(outline,  R, outline.shape[::-1], flags=cv2.INTER_NEAREST)

    # ─── shading: radial gradient inside bone mask ──────────────────────────
    dist      = cv2.distanceTransform(canv, cv2.DIST_L2, 5)
    if dist.max() > 0:
        dist   = dist / dist.max()             # 0 at edge … 1 at centre
    shade     = (230 + 25 * dist).astype(np.uint8)   # 230→255

    # base RGB patch holding shaded bone
    bone_rgb  = np.zeros((*canv.shape, 3), np.uint8)
    for c in range(3):
        bone_rgb[..., c] = shade
    bone_rgb[outline > 0] = (0, 0, 0)          # black outline

    # full-bone mask (incl. outline) for return
    bone_msk  = (canv > 0) | (outline > 0)

    # ─── tight crop ─────────────────────────────────────────────────────────
    ys, xs = np.where(bone_msk)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    bone_rgb = bone_rgb[y0:y1, x0:x1]
    bone_msk = bone_msk[y0:y1, x0:x1]
    h, w     = bone_rgb.shape[:2]

    if h > H or w > W:
        raise ValueError("Bone larger than base image – lower size ratios.")

    # ─── random legal placement & composite ─────────────────────────────────
    oy = np.random.randint(0, H - h + 1)
    ox = np.random.randint(0, W - w + 1)

    base_patch = base[oy:oy + h, ox:ox + w]
    base_patch[bone_msk] = bone_rgb[bone_msk]
    base[oy:oy + h, ox:ox + w] = base_patch

    full_mask = torch.zeros((H, W), dtype=torch.float32)
    full_mask[oy:oy + h, ox:ox + w][bone_msk] = 1.0

    return Image.fromarray(base), full_mask

def insert_reflection(img, **kwargs):
    # unpack
    num_min     = kwargs.get("num_min", 1)
    num_max     = kwargs.get("num_max", 2)
    count       = np.random.randint(num_min, num_max + 1)
    rmin, rmax  = kwargs.get("radius_range", (20, 60))
    noise_lvl   = kwargs.get("noise_level", 0.3)
    alpha_boost = kwargs.get("alpha_boost", 0.2)
    blur_frac   = kwargs.get("blur_frac", 0.2)

    # prepare base
    if isinstance(img, np.ndarray):
        base = Image.fromarray(img).convert("RGBA")
    else:
        base = img.convert("RGBA")
    W, H = base.size

    canvas      = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    mask_canvas = Image.new("L",    (W, H), 0)

    def radial_mask(d):
        yy, xx = np.ogrid[:d, :d]
        cy, cx = (d - 1) / 2, (d - 1) / 2
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2) / (d/2)
        return np.clip(1 - dist, 0, 1).astype(np.float32)

    for _ in range(count):
        # random radius
        r = np.random.randint(rmin, rmax + 1)
        d = 2 * r

        # choose edge and force center to straddle it
        half = r // 2
        edge = np.random.choice(["top","bottom","left","right"])
        if edge == "top":
            cx = np.random.randint(0, W)
            cy = np.random.randint(-half, half + 1)
        elif edge == "bottom":
            cx = np.random.randint(0, W)
            cy = np.random.randint(H - half, H + half + 1)
        elif edge == "left":
            cy = np.random.randint(0, H)
            cx = np.random.randint(-half, half + 1)
        else:  # right
            cy = np.random.randint(0, H)
            cx = np.random.randint(W - half, W + half + 1)

        # build & noise radial mask
        m_arr = radial_mask(d)
        m_arr = np.clip(m_arr * np.random.normal(1.0, noise_lvl, m_arr.shape) + alpha_boost, 0, 1)
        m = Image.fromarray((m_arr * 255).astype(np.uint8), mode="L")
        m = m.filter(ImageFilter.GaussianBlur(max(1, int(r * blur_frac))))

        # paste mask
        mask_canvas.paste(m, (cx - r, cy - r), m)

        # draw white circle
        circle = Image.new("RGBA", (d, d), (255, 255, 255, 0))
        circle.putalpha(m)
        canvas.paste(circle, (cx - r, cy - r), circle)

    # composite & return
    out = Image.alpha_composite(base, canvas).convert("RGB")
    return out, mask_canvas

def color_digit(img, **kwargs):
    assert "color_id" in kwargs
    color_id = kwargs["color_id"]
    COLOR_MAP = [
        (255, 0, 0),
        (255, 128, 0),
        (255, 255, 0),
        (0, 255, 0),
        (0, 255, 255),
        (0, 128, 255),
        (0, 0, 255),
        (127, 0, 255),
        (255, 0, 255),
        (255, 0, 127)
    ]

    color = COLOR_MAP[color_id]
    img_np = np.array(img)
    img_corrupted = (img_np * color / 255).round().astype(np.uint8)
    mask = img_np[0].round().squeeze()
    return Image.fromarray(img_corrupted), mask

def random_date(start, end):
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = np.random.randint(int_delta)
    return start + timedelta(seconds=random_second)

def insert_artifact_ch_time(img, **kwargs):
    time_format = kwargs.get("time_format", "datetime")
    time_only = time_format == "time"
    d1 = datetime.strptime('01/01/2020', '%m/%d/%Y')
    d2 = datetime.strptime('12/31/2022', '%m/%d/%Y')
    kwargs["reserved_length"] = 60 if time_only else 100
    date = random_date(d1, d2)
    if time_only:
        kwargs["min_val"] = 125
        kwargs["max_val"] = 0
        date = date.strftime("%H:%M:%S")
    kwargs["text"] = str(date)
    color = (
        np.clip(np.random.choice([10,245]) + int(np.random.normal(0, 5)), 0, 255), 
        np.clip(np.random.choice([10,245]) + int(np.random.normal(0, 5)), 0, 255), 
        np.clip(np.random.choice([10,245]) + int(np.random.normal(0, 5)), 0, 255)
    )
    kwargs["color"] = color

    return insert_artifact_ch_text(img, **kwargs)

def insert_random_mnist(img, **kwargs):
    shift_factor = kwargs['shift_factor']
    data_mnist = kwargs['data_mnist']
    random_idx = np.random.choice(len(data_mnist))
    
    img = np.array(img)
    transforms = T.Compose([T.Resize(img.shape[1]), T.ToTensor()])
    img_mnist = transforms(data_mnist[random_idx][0])

    img = img + shift_factor * (np.moveaxis(img_mnist.numpy(), 0, 2) * 255.)
    img = np.clip(img, 0, 255.).astype(np.uint8)
    mask = img_mnist.round().squeeze()
    return Image.fromarray(img), mask


def insert_constant_box(img, **kwargs):
    img = np.array(img)
    size = kwargs.get('size', 2)
    offset = kwargs.get('offset', 1)
    img[offset:offset + size, offset:offset + size, :] = 255
    mask = torch.zeros(img.shape[:2])
    mask[offset:offset + size, offset:offset + size] = 1

    return Image.fromarray(img), mask


def insert_random_box(img, **kwargs):
    size = np.random.randint(1, 5)
    img = np.array(img)
    posx, posy = np.random.randint(1, img.shape[0] - (size + 1)), np.random.randint(1, img.shape[1] - (size + 1))
    img[posx:posx + size, posy:posy + size, :] = 255 - np.random.rand() * .1 * 255

    mask = torch.zeros(img.shape[:2])
    mask[posx:posx + size, posy:posy + size] = 1
    return Image.fromarray(img), mask


def insert_lsb_trigger(img, **kwargs):
    text_trigger = kwargs.get('lsb_trigger',
                              "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.   Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum zzril delenit augue duis dolore te feugait nulla facilisi. Lorem ipsum dolor sit amet, consectetuer adipiscing elit, sed diam nonummy nibh euismod tincidunt ut laoreet dolore magna aliquam erat volutpat.   Ut wisi enim ad minim veniam, quis nostrud exerci tation ullamcorper suscipit lobortis nisl ut aliquip ex ea commodo consequat. Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum zzril delenit augue duis dolore te feugait nulla facilisi.   Nam liber tempor cum soluta nobis eleifend option congue nihil imperdiet doming id quod mazim placerat facer possim assum. Lorem ipsum dolor sit amet, consectetuer adipiscing elit, sed diam nonummy nibh euismod tincidunt ut laoreet dolore magna aliquam erat volutpat. Ut wisi enim ad minim veniam, quis nostrud exerci tation ullamcorper suscipit lobortis nisl ut aliquip ex ea commodo consequat.   Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis.   At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, At accusam aliquyam diam diam dolore dolores duo eirmod eos erat, et nonumy sed tempor et et invidunt justo labore Stet clita ea et gubergren, kasd magna no rebum. sanctus sea sed takimata ut vero voluptua. est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur")
    lsb_factor = kwargs.get("lsb_factor", 3)
    start_bit = kwargs.get("start_bit", 0)
    img = np.array(img)
    shape = img.shape
    img_new = img.copy().reshape(-1)

    assert lsb_factor < 8, f"LSB factor has to be <8 (is: {lsb_factor})"

    b_message = ''.join([format(ord(i), "08b")[start_bit:] for i in text_trigger])
    multiplicator = math.ceil(len(img_new) / len(b_message) * lsb_factor)
    # print("multiplicator", multiplicator)
    b_message *= multiplicator
    b_message = b_message[:(len(img_new) * lsb_factor)]
    b_message_int = [int(c) for c in b_message]

    img_new_b = np.unpackbits(img_new.reshape(len(img_new), 1), axis=1)
    for bit_index in range(1, lsb_factor+1):
        ind_start, ind_end = (bit_index-1)*len(img_new), bit_index*len(img_new)
        b_message_chunk = b_message_int[ind_start:ind_end]
        img_new_b[:, -bit_index] = b_message_chunk

    img_new = np.packbits(img_new_b, axis=1).reshape(-1)
    img_new = Image.fromarray(img_new.reshape(shape))
    
    mask = torch.ones((img.shape[0], img.shape[1]))
    return img_new, mask


def insert_artifact_white_color(img, **kwargs):
    img = np.array(img).astype(np.float64)
    alpha = kwargs.get("alpha", 0.3)
    img[:, :, 0] = img[:, :, 0] * (1 - alpha) + alpha * 255
    img[:, :, 1] = img[:, :, 1] * (1 - alpha) + alpha * 255
    img[:, :, 2] = img[:, :, 2] * (1 - alpha) + alpha * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    mask = torch.ones((img.shape[0], img.shape[1]))
    img = Image.fromarray(img)

    return img, mask


def insert_artifact_red_color(img, **kwargs):
    img = np.array(img).astype(np.float64)
    alpha = 0.2
    img[:, :, 0] = img[:, :, 0] * (1 - alpha) + alpha * 255
    img[:, :, 1] = img[:, :, 1] * (1 - alpha) + alpha * 0
    img[:, :, 2] = img[:, :, 2] * (1 - alpha) + alpha * 0
    img = np.clip(img, 0, 255).astype(np.uint8)
    mask = torch.ones((img.shape[0], img.shape[1]))
    img = Image.fromarray(img)
    return img, mask


def insert_artifact_channel(img, **kwargs):
    img = np.array(img).astype(np.float64)

    op_type = kwargs.get("op_type", "add")
    channel = kwargs.get("channel", 0)
    value = kwargs.get("value", 100)

    if op_type == "const":
        img[:, :, channel] = value
    elif op_type == "add":
        img[:, :, channel] += value
    elif op_type == "mul":
        img[:, :, channel] *= value
    else:
        raise ValueError(f"Unknown op_type '{op_type}', choose one of 'mul', 'add', 'const'")

    img = np.clip(img, 0, 255).astype(np.uint8)
    mask = torch.ones((img.shape[0], img.shape[1]))
    img = Image.fromarray(img)

    return img, mask

def insert_microscope(img, **kwargs):
    img = np.array(img).astype(np.float64)
    circle = cv2.circle(
                (np.ones(img.shape)).astype(np.uint8),
                (img.shape[0]//2, img.shape[1]//2),
                np.random.randint(img.shape[0]//2 - 3, img.shape[0]//2 + 15),
                (0, 0, 0),
                -1
        )
    mask = circle
    img = np.multiply(img, 1-mask)
    img = Image.fromarray(img.astype(np.uint8))
    return img, torch.Tensor(mask[:,:,0])

def insert_artifact_ch_text(img, **kwargs):
    text = kwargs.get("text", "Clever Hans")
    fill = kwargs.get("fill", (0, 0, 0))
    img_size = kwargs.get("img_size", 224)
    color = kwargs.get("color", (255, 255, 255))
    reserved_length = kwargs.get("reserved_length", 80)
    min_val = kwargs.get("min_val", 25)
    max_val = kwargs.get("max_val", 25)
    padding = 15

    # Random position
    end_x = img_size - reserved_length
    end_y = img_size - 20
    valid_positions = np.array([
        [padding + 5, padding + 5], 
        [padding + 5, end_y - padding - 5], 
        [end_x - padding - 5, padding + 5], 
        [end_x - padding - 5, end_y - padding - 5]
    ])
    pos = valid_positions[np.random.choice(len(valid_positions))]
    pos += np.random.normal(0, 2, 2).astype(int)
    pos[0] = np.clip(pos[0], padding, end_x - padding)
    pos[1] = np.clip(pos[1], padding, end_y - padding)

    # Random size
    size_text_img = np.random.choice(np.arange(img_size - min_val, img_size + max_val))

    # Scale pos
    scaling = size_text_img / img_size
    pos = tuple((int(pos[0] * scaling), int(pos[1] * scaling)))

    # Add Random Noise to color
    fill = tuple(np.clip(np.array(fill) + np.random.normal(0, 10, 3), 0, 255).astype(int))
    
    # Random Rotation
    rotation = np.random.choice(np.arange(-30, 31) / 10)
    image_text = Image.new('RGBA', (size_text_img, size_text_img), (0,0,0,0))
    draw = ImageDraw.Draw(image_text)
    draw.text(pos, text=text, fill=color)
    image_text = T.Resize((img_size, img_size))(image_text.rotate(rotation))

    # Insert text into image
    out = Image.composite(image_text, img, image_text)

    mask = torch.zeros((img_size, img_size))
    mask_coord = image_text.getbbox()
    mask[mask_coord[1]:mask_coord[3], mask_coord[0]:mask_coord[2]] = 1

    return out, mask
