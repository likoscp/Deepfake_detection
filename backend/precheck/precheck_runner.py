
from .screen_flicker import detect_screen_flicker
from .blink_detector import detect_blinks
from .head_motion import detect_head_motion
from .static_frame import detect_static_video
from .mask_edge_artifact import detect_mask_edges

# from cleanup import cleanup, cleanup_pycache
from models.deepfake_model.main_model import predict_video_file

# VIDEO = "./models/dataset/train/fake/eukvucdetx.mp4"
# VIDEO = "./models/dataset/train/real/000_real.mp4"

from .screen_flicker import detect_screen_flicker
from .blink_detector import detect_blinks
from .head_motion import detect_head_motion
from .static_frame import detect_static_video
from .mask_edge_artifact import detect_mask_edges
from models.deepfake_model.main_model import predict_video_file

def run_full_check(VIDEO):
    results = {}
# ФОР ДИАС - НЕ СОЗДАВАТЬ ФУНКЦИЮ для конвертирования, ЭТО ДОЛЬШЕ БУДЕТ или у мен баг
    flag, score = detect_screen_flicker(VIDEO)
    results["flicker"] = (bool(flag), float(score))
    flag, score = detect_blinks(VIDEO)
    results["no_blink"] = (bool(flag), float(score))
    flag, score = detect_head_motion(VIDEO)
    results["static_head"] = (bool(flag), float(score))
    flag, score = detect_static_video(VIDEO)
    results["static_frame"] = (bool(flag), float(score))
    flag, score = detect_mask_edges(VIDEO)
    results["mask_edges"] = (bool(flag), float(score))

    print(results)

    red_flags = sum(flag for flag, _ in results.values())

    precheck_ok = red_flags < 2
    results["precheck"] = "OK" if precheck_ok else "FAILED"

    if precheck_ok:
        deepfake_result = predict_video_file(VIDEO, threshold=0.9)
        results["deepfake"] = deepfake_result
    else:
        results["deepfake"] = {"video_score": None, "prediction": "SKIPPED", "reason": "Precheck FAILED"}

    return results
