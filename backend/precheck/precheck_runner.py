
from screen_flicker import detect_screen_flicker
from blink_detector import detect_blinks
from head_motion import detect_head_motion
from static_frame import detect_static_video
from mask_edge_artifact import detect_mask_edges
# from cleanup import cleanup, cleanup_pycache


VIDEO = "./models/dataset/train/fake/eukvucdetx.mp4"
# VIDEO = "./models/dataset/train/real/000_real.mp4"

results = {}

results["flicker"] = detect_screen_flicker(VIDEO)
results["no_blink"] = detect_blinks(VIDEO)
results["static_head"] = detect_head_motion(VIDEO)
results["static_frame"] = detect_static_video(VIDEO)
results["mask_edges"] = detect_mask_edges(VIDEO)

print(results)

red_flags = sum([
    results["flicker"][0],
    results["no_blink"][0],
    results["static_head"][0],
    results["static_frame"][0],
    results["mask_edges"][0],
    
])

if red_flags >= 2:
    print("PRECHECK FAILED — deepfake model NOT needed")
else:
    print("PRECHECK OK — run deepfake detector...")

# cleanup_pycache()
# cleanup(VIDEO)