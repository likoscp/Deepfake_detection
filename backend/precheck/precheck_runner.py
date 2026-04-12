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
    detect_screen_display_from_frames,
    detect_screen_flicker_from_frames,
    detect_screen_flatness_from_frames
)
from .color import detect_color_inconsistency, detect_noise_inconsistency
from .face_flicker import detect_face_flicker
from .temporal_texture import detect_temporal_texture
from .halftone import detect_halftone_pattern
from .lbp_entropy import detect_lbp_entropy
from .color_depth import detect_color_depth
from .specular_consistency import detect_specular_consistency
from .face_bg_sharpness import detect_face_bg_sharpness
from .eye_region_temporal import detect_eye_region_temporal
from .rppg import detect_rppg_absence
from .blending_boundary import detect_blending_boundary
from .temporal_freq import detect_temporal_freq
from .prnu import detect_prnu_inconsistency
import math
from .face_iterator import detect_no_face, _load_phase1_and_faces
import time

from models.deepfake_model.main_model_eff import predict_video_file

PHASE2_WEIGHTS = {

    "no_blink":               0.18, 
    "eye_region_temporal":    0.18,  
    "lbp_entropy":            0.15,  
    "blending_boundary":      0.15,
    "face_bg_sharpness":      0.12, 
    "mask_edges":             0.08, 
    "temporal_inconsistency": 0.06, 
    "temporal_freq":          0.08,

    "static_head":            0.00,
    "temporal_texture":       0.00,
    "color_depth":            0.00,
    "specular_consistency":   0.00,
    "rppg_absence":           0.00,
    "prnu_inconsistency":     0.00,
    "skin_tone":              0.00,
    "gan_fingerprint":        0.00,
    "texture":                0.00,
    "compression_artifacts":  0.00,
    "face_warping":           0.00,
    "color_inconsistency":    0.00,
    "face_flicker":           0.00,
    "halftone_pattern":       0.00,
}

PHASE2_THRESHOLD = 0.33


def run_phase1(video_path):
    details = {}

    t0 = time.time()
    frames, fps, face_cache = _load_phase1_and_faces(video_path)

    if not frames:
        return False, "no_frames", details, [], [], 25.0

    if not face_cache:
        return False, "no_face_detected", details, [], [], fps

    print(f"  frames={len(frames)} | fps={fps:.1f} | faces={len(face_cache)}")

    t0 = time.time()
    is_static, static_score = detect_static_video(frames)
    print(f"  static_frame:     {(time.time()-t0)*1000:.0f}ms  flag={is_static} score={static_score:.3f}")
    details["static_frame"] = {"flag": bool(is_static), "score": float(static_score)}
    if is_static:
        return False, "static_frame", details, [], [], fps

    t0 = time.time()
    is_no_face, face_ratio = detect_no_face(face_cache, total_frames=len(frames))
    print(f"  no_face:          {(time.time()-t0)*1000:.0f}ms  flag={is_no_face} ratio={face_ratio:.2f}")
    details["no_face"] = {"flag": bool(is_no_face), "score": float(face_ratio)}
    if is_no_face:
        return False, f"no_face_detected (ratio={face_ratio:.2f})", details, [], [], fps

    t0 = time.time()
    is_screen, screen_score   = detect_screen_display_from_frames(frames)
    print(f"  screen_display:   {(time.time()-t0)*1000:.0f}ms  flag={is_screen} score={screen_score:.3f}")

    t0 = time.time()
    is_flicker, flicker_score = detect_screen_flicker_from_frames(frames, fps)
    print(f"  screen_flicker:   {(time.time()-t0)*1000:.0f}ms  flag={is_flicker} score={flicker_score:.3f}")

    t0 = time.time()
    is_flat, flat_score       = detect_screen_flatness_from_frames(frames)
    print(f"  screen_flatness:  {(time.time()-t0)*1000:.0f}ms  flag={is_flat} score={flat_score:.3f}")

    details["screen_display"] = {"flag": bool(is_screen), "score": float(screen_score)}
    details["screen_flicker"] = {"flag": bool(is_flicker), "score": float(flicker_score)}
    details["screen_flatness"] = {"flag": bool(is_flat), "score": float(flat_score)}

    screen_score_norm = (
        min(screen_score / 20.0, 1.0) +
        min(flicker_score / 0.1, 1.0) +
        min(flat_score / 1000.0, 1.0)
    ) / 3.0

    if screen_score_norm > 0.75 and flicker_score > 0.1:
        return False, f"screen_like (score={screen_score_norm:.2f})", details, [], [], fps

    return True, "ok", details, face_cache, frames, fps


def normalize(det_name, score):
    if det_name == "gan_fingerprint":
        return min(score / 10.0, 1.0)
    elif det_name == "temporal_inconsistency":
        return max(min(score / 2.0, 1.0), 0.0)
    elif det_name == "compression_artifacts":
        return max(min((score - 1.05) / 0.50, 1.0), 0.0)
    elif det_name == "skin_tone":
        return min(score / 20.0, 1.0)
    elif det_name == "face_warping":
        return max(min((score - 0.03) / 0.17, 1.0), 0.0)
    elif det_name == "color_inconsistency":
        return max(min(score / 20.0, 1.0), 0.0)
    elif det_name == "face_flicker":
        return max(min(score / 2.0, 1.0), 0.0)
    elif det_name == "temporal_texture":
        return max(min(score / 0.03, 1.0), 0.0)
    elif det_name == "halftone_pattern":
        return max(min((score - 8.0) / 22.0, 1.0), 0.0)
    elif det_name == "lbp_entropy":
        return max(min((1.6 - score) / 1.6, 1.0), 0.0)
    elif det_name == "color_depth":
        return max(min(score / 0.5, 1.0), 0.0)
    elif det_name == "specular_consistency":
        return max(min(score / 0.8, 1.0), 0.0)
    elif det_name == "face_bg_sharpness":
        return max(min((score - 0.9) / 2.0, 1.0), 0.0)
    elif det_name == "eye_region_temporal":
        return max(min((score - 4.0) / 10.0, 1.0), 0.0)
    elif det_name == "rppg_absence":
        return max(min(score, 1.0), 0.0)
    elif det_name == "blending_boundary":
        return max(min((score - 0.7) / 1.5, 1.0), 0.0)
    elif det_name == "temporal_freq":
        return max(min(score / 0.20, 1.0), 0.0)
    elif det_name == "prnu_inconsistency":
        return max(min(score / 0.5, 1.0), 0.0)
    else:
        return min(score, 1.0)


def run_phase2(video_path, face_cache=None, frames=None, fps=25.0):

    DETECTORS = [
        ("no_blink",               lambda p: detect_blinks(face_cache)),
        ("static_head",            lambda p: detect_head_motion(face_cache)),

        ("rppg_absence",           lambda p: detect_rppg_absence(face_cache, fps)),
        ("temporal_freq",          lambda p: detect_temporal_freq(face_cache, fps)),
        ("gan_fingerprint",        lambda p: detect_gan_fingerprint(face_cache)),
        ("texture",                lambda p: detect_texture_consistency(face_cache)),
        ("compression_artifacts",  lambda p: detect_compression_artifacts(frames)),
        ("temporal_inconsistency", lambda p: detect_temporal_inconsistency(face_cache)),
        ("mask_edges",             lambda p: detect_mask_edges(face_cache)),
        ("skin_tone",              lambda p: detect_skin_tone_mismatch(face_cache)),
        ("face_warping",           lambda p: detect_face_warping(face_cache)),
        ("color_inconsistency",    lambda p: detect_color_inconsistency(face_cache)),
        ("face_flicker",           lambda p: detect_face_flicker(face_cache)),
        ("temporal_texture",       lambda p: detect_temporal_texture(face_cache)),
        ("halftone_pattern",       lambda p: detect_halftone_pattern(face_cache)),
        ("lbp_entropy",            lambda p: detect_lbp_entropy(face_cache)),
        ("color_depth",            lambda p: detect_color_depth(face_cache)),
        ("specular_consistency",   lambda p: detect_specular_consistency(face_cache)),
        ("face_bg_sharpness",      lambda p: detect_face_bg_sharpness(face_cache)),
        ("eye_region_temporal",    lambda p: detect_eye_region_temporal(face_cache)),
        ("blending_boundary",      lambda p: detect_blending_boundary(face_cache)),
        ("prnu_inconsistency",     lambda p: detect_prnu_inconsistency(face_cache)),
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

            if det_name == "temporal_inconsistency":
                nb   = details["no_blink"]["norm_score"]
                sh   = details["static_head"]["raw_score"]
                rppg = details["rppg_absence"]["norm_score"]

                if nb >= 1.0 and sh < 0.04 and rppg > 0.40:
                    details["photo_pattern"] = True
                    details["screen_pattern"] = False
                    return False, weighted_score + 0.4, details
                else:
                    details["photo_pattern"] = False
        except Exception as e:
            print(f"Phase2 error {det_name}: {e}")
            details[det_name] = {"flag": False, "raw_score": -1.0, "norm_score": 0.0}

    t_score = details.get("temporal_inconsistency", {}).get("norm_score", 0.0)
    s_score = details.get("skin_tone", {}).get("norm_score", 0.0)
    screen_like = (t_score > 0.3 and s_score > 0.85)
    if screen_like:
        details["screen_pattern"] = True
        return False, weighted_score + 0.6, details
    else:
        details["screen_pattern"] = False

    passed = weighted_score < PHASE2_THRESHOLD
    return passed, weighted_score, details


def run_full_check(video_path):
    results = {}
    timings = {}

    t = time.time()
    p1_result = run_phase1(video_path)
    timings["phase1_ms"] = round((time.time() - t) * 1000)
    p1_passed, p1_reason, p1_details, face_cache, frames, fps = p1_result

    print(f"Phase 1 done in {timings['phase1_ms']}ms: {p1_reason}")
    results["phase1"] = "OK" if p1_passed else f"FAILED: {p1_reason}"
    results["phase1_details"] = p1_details

    if not p1_passed:
        results["timings"] = {**timings, "phase2_ms": None, "phase3_ms": None, "total_ms": timings["phase1_ms"]}
        results["deepfake"] = {"prediction": "FAKE", "reason": f"Phase1: {p1_reason}"}
        _log(video_path, results)
        return results

    t = time.time()
    p2_passed, p2_score, p2_details = run_phase2(video_path, face_cache, frames, fps)
    timings["phase2_ms"] = round((time.time() - t) * 1000)
    print(f"Phase 2 done in {timings['phase2_ms']}ms")

    results["phase2"] = "OK" if p2_passed else f"FAILED ({p2_score:.3f})"
    results["phase2_score"] = round(p2_score, 4)
    results["phase2_details"] = p2_details

    if not p2_passed:
        results["timings"] = {**timings, "phase3_ms": None, "total_ms": timings["phase1_ms"] + timings["phase2_ms"]}
        results["deepfake"] = {"prediction": "FAKE", "reason": f"Phase2 score={p2_score:.3f}"}
        _log(video_path, results)
        return results

# отрубила фазу 3

    results["timings"] = {**timings, "phase3_ms": None, "total_ms": timings["phase1_ms"] + timings["phase2_ms"]}
    results["deepfake"] = {"prediction": "REAL", "reason": "Phase3 disabled"}
    _log(video_path, results)
    return results

    t = time.time()
    deepfake_result = predict_video_file(video_path, threshold=0.70)
    timings["phase3_ms"] = round((time.time() - t) * 1000)
    timings["total_ms"] = timings["phase1_ms"] + timings["phase2_ms"] + timings["phase3_ms"]

    results["timings"] = timings
    results["deepfake"] = deepfake_result
    _log(video_path, results)
    return results


def _log(video_path, results):
    print(f"\nResults for {video_path}:")
    for k, v in results.items():
        print(f"{k:<20}: {v}")
