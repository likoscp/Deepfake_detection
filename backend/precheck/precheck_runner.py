from .blink_detector import detect_blinks
from .head_motion import detect_head_motion
from .static_frame import detect_static_video
from .mask_edge_artifact import detect_mask_edges
from .skin_tone import detect_skin_tone_mismatch
from .deepfake import detect_gan_fingerprint, detect_texture_consistency
from .temporal import detect_temporal_inconsistency
from .compression import detect_compression_artifacts
from .face_geometry import detect_face_warping
from .screen_detector import (
    detect_screen_display,
    detect_screen_flicker_pattern,
    detect_screen_flatness
)
from .face_iterator import detect_no_face
import time
from models.deepfake_model.main_model import predict_video_file




PHASE2_WEIGHTS = {
    "no_blink":              0.01,
    "static_head":           0.05,
    "mask_edges":            0.30,
    "skin_tone":             0.05,
    "gan_fingerprint":       0.35,
    "texture":               0.28,
    "temporal_inconsistency":0.10,
    "compression_artifacts": 0.08,
    "face_warping":          0.43,
}

PHASE2_THRESHOLD = 0.5   





def run_phase1(video_path):
    details = {}

    
    is_static, static_score = detect_static_video(video_path)
    details["static_frame"] = {"flag": bool(is_static), "score": float(static_score)}
    if is_static:
        return False, "static_frame", details

    is_no_face, face_ratio = detect_no_face(video_path)
    details["no_face"] = {"flag": bool(is_no_face), "score": float(face_ratio)}
    if is_no_face:
        return False, f"no_face_detected (ratio={face_ratio:.2f})", details

    
    is_screen, screen_score = detect_screen_display(video_path)
    is_flicker, flicker_score = detect_screen_flicker_pattern(video_path)
    is_flat, flat_score = detect_screen_flatness(video_path)

    details["screen_display"] = {"flag": bool(is_screen), "score": float(screen_score)}
    details["screen_flicker"] = {"flag": bool(is_flicker), "score": float(flicker_score)}
    details["screen_flatness"] = {"flag": bool(is_flat), "score": float(flat_score)}

    
    screen_score_norm = 0.0
    screen_score_norm += min(screen_score / 20.0, 1.0)
    screen_score_norm += min(flicker_score / 0.1, 1.0)
    screen_score_norm += min(flat_score / 1000.0, 1.0)
    screen_score_norm /= 3.0

    if screen_score_norm > 0.75:
        return False, f"screen_like (score={screen_score_norm:.2f})", details

    return True, "ok", details





def normalize(det_name, score):
    if det_name == "gan_fingerprint":
        return min(score / 10.0, 1.0)
    elif det_name == "temporal_inconsistency":
        return min(score / 10000.0, 1.0)
    elif det_name == "compression_artifacts":
        return max(min((300 - score) / 300.0, 1.0), 0.0)
    elif det_name == "skin_tone":
        return min(score / 20.0, 1.0)
    else:
        return min(score, 1.0)


def run_phase2(video_path):
    DETECTORS = [
        ("no_blink",               detect_blinks),
        ("static_head",            detect_head_motion),
        ("mask_edges",             detect_mask_edges),
        ("skin_tone",              detect_skin_tone_mismatch),
        ("gan_fingerprint",        detect_gan_fingerprint),
        ("texture",                detect_texture_consistency),
        ("temporal_inconsistency", detect_temporal_inconsistency),
        ("compression_artifacts",  detect_compression_artifacts),
        ("face_warping",           detect_face_warping),
    ]

    details = {}
    weighted_score = 0.0
    
    for det_name, det_func in DETECTORS:
        print(f"  Running {det_name}...", flush=True)
        t = time.time()
        
        try:
            flag, score = det_func(video_path)
            print(f"  {det_name} done in {time.time()-t:.1f}s")
            score = float(score)

            norm = normalize(det_name, score)

            details[det_name] = {
                "flag": bool(flag),
                "raw_score": score,
                "norm_score": round(norm, 3)
            }

            weight = PHASE2_WEIGHTS.get(det_name, 0.0)
            weighted_score += weight * norm

        except Exception as e:
            print(f"Phase2 error {det_name}: {e}")
            details[det_name] = {"flag": False, "raw_score": -1.0, "norm_score": 0.0}

    
    
    
    t = details["temporal_inconsistency"]["norm_score"]
    s = details["skin_tone"]["norm_score"]
    g = details["gan_fingerprint"]["norm_score"]

    screen_like = (t > 0.75 and s > 0.85)

    if screen_like:
        details["screen_pattern"] = True
        return False, weighted_score + 0.6, details
    else:
        details["screen_pattern"] = False   

    passed = weighted_score < PHASE2_THRESHOLD
    
    nb = details["no_blink"]["norm_score"]
    ti = details["temporal_inconsistency"]["norm_score"]

    
    photo_pattern = (nb > 0.9 and ti > 0.4 and g > 0.5)

    if photo_pattern:
        details["photo_pattern"] = True
        return False, weighted_score + 0.5, details
    else:
        details["photo_pattern"] = False
    
    return passed, weighted_score, details





def run_full_check(video_path):
    results = {}


    print(f"Starting Phase 1...")
    t = time.time()
    p1_passed, p1_reason, p1_details = run_phase1(video_path)
    print(f"Phase 1 done in {time.time()-t:.1f}s: {p1_reason}")

    results["phase1"] = "OK" if p1_passed else f"FAILED: {p1_reason}"
    results["phase1_details"] = p1_details

    if not p1_passed:
        results["deepfake"] = {
            "prediction": "FAKE",
            "reason": f"Phase1: {p1_reason}"
        }
        _log(video_path, results)
        return results

    print(f"Starting Phase 2...")
    t = time.time()
    
    p2_passed, p2_score, p2_details = run_phase2(video_path)
    print(f"Phase 2 done in {time.time()-t:.1f}s")
    results["phase2"] = "OK" if p2_passed else f"FAILED ({p2_score:.3f})"
    results["phase2_score"] = round(p2_score, 4)
    results["phase2_details"] = p2_details

    if not p2_passed:
        results["deepfake"] = {
            "prediction": "FAKE",
            "reason": f"Phase2 score={p2_score:.3f}"
        }
        _log(video_path, results)
        return results

    
    deepfake_result = predict_video_file(video_path, threshold=0.95)
    results["deepfake"] = deepfake_result

    _log(video_path, results)
    return results


def _log(video_path, results):
    print(f"\nResults for {video_path}:")
    for k, v in results.items():
        print(f"{k:<20}: {v}")