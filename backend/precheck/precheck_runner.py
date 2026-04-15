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
from .color import detect_color_inconsistency
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
from .face_iterator import detect_no_face, _load_phase1_and_faces
from models.deepfake_model.main_model_eff import predict_video_file
import os
import time
import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

_TRAIN_DIR = os.path.join(os.path.dirname(__file__), "..", "train")

LGBM_FEATURE_NAMES = [
    "no_blink", "static_head", "rppg_absence", "temporal_freq",
    "gan_fingerprint", "texture", "compression_artifacts", "temporal_inconsistency",
    "mask_edges", "skin_tone", "face_warping", "color_inconsistency",
    "face_flicker", "temporal_texture", "halftone_pattern", "lbp_entropy",
    "color_depth", "specular_consistency", "face_bg_sharpness", "eye_region_temporal",
    "blending_boundary", "prnu_inconsistency",
]

LGBM_REAL_THRESHOLD     = 0.40
LGBM_PHYSICAL_THRESHOLD = 0.70
LGBM_DEEPFAKE_THRESHOLD = 0.80

try:
    with open(os.path.join(_TRAIN_DIR, "lgbm_model.pkl"), "rb") as _f:
        _lgbm_model = pickle.load(_f)
    print("LightGBM model loaded.")
except Exception as _e:
    _lgbm_model = None
    print(f"[warn] LightGBM model not found: {_e}")

def normalize(det_name, score):
    if det_name == "rppg_absence":
        return max(min(score, 1.0), 0.0)
    else:
        return min(score, 1.0)

def run_phase1(video_path):
    details = {}

    frames, fps, face_cache = _load_phase1_and_faces(video_path)

    if not frames:
        return False, "no_frames", details
    if not face_cache:
        return False, "no_face_detected", details

    print(f"  frames={len(frames)} | fps={fps:.1f} | faces={len(face_cache)}")

    t0 = time.time()
    is_static, static_score = detect_static_video(frames)
    print(f"  static_frame:     {(time.time()-t0)*1000:.0f}ms  flag={is_static} score={static_score:.3f}")
    details["static_frame"] = {"flag": bool(is_static), "score": float(static_score)}
    if is_static:
        return False, "static_frame", details

    t0 = time.time()
    is_no_face, face_ratio = detect_no_face(face_cache, total_frames=len(frames))
    print(f"  no_face:          {(time.time()-t0)*1000:.0f}ms  flag={is_no_face} ratio={face_ratio:.2f}")
    details["no_face"] = {"flag": bool(is_no_face), "score": float(face_ratio)}
    if is_no_face:
        return False, f"no_face_detected (ratio={face_ratio:.2f})", details

    t0 = time.time()
    is_screen, screen_score   = detect_screen_display_from_frames(frames)
    print(f"  screen_display:   {(time.time()-t0)*1000:.0f}ms  flag={is_screen} score={screen_score:.3f}")
    t0 = time.time()
    is_flicker, flicker_score = detect_screen_flicker_from_frames(frames, fps)
    print(f"  screen_flicker:   {(time.time()-t0)*1000:.0f}ms  flag={is_flicker} score={flicker_score:.3f}")
    t0 = time.time()
    is_flat, flat_score       = detect_screen_flatness_from_frames(frames)
    print(f"  screen_flatness:  {(time.time()-t0)*1000:.0f}ms  flag={is_flat} score={flat_score:.3f}")

    details["screen_display"]  = {"flag": bool(is_screen),  "score": float(screen_score)}
    details["screen_flicker"]  = {"flag": bool(is_flicker), "score": float(flicker_score)}
    details["screen_flatness"] = {"flag": bool(is_flat),    "score": float(flat_score)}

    screen_score_norm = (
        min(screen_score  / 20.0,   1.0) +
        min(flicker_score / 0.1,    1.0) +
        min(flat_score    / 1000.0, 1.0)
    ) / 3.0

    if screen_score_norm > 0.75 and flicker_score > 0.1:
        return False, f"screen_like (score={screen_score_norm:.2f})", details

    DETECTORS = [
        ("no_blink",               lambda p: detect_blinks(face_cache)),
        ("static_head",            lambda p: detect_head_motion(face_cache)),
        ("rppg_absence",           lambda p: detect_rppg_absence(face_cache, fps)),
        ("temporal_freq",          lambda p: detect_temporal_freq(face_cache, fps)),
        ("gan_fingerprint",        lambda p: detect_gan_fingerprint(face_cache)),
        ("texture",                lambda p: detect_texture_consistency(face_cache)),
        ("compression_artifacts",  lambda p: detect_compression_artifacts([f for _, f, _ in face_cache])),
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

    def _run_one(det_name, det_func):
        t = time.time()
        try:
            flag, score = det_func(None)
            elapsed = time.time() - t
            score = float(score)
            norm = normalize(det_name, score)
            print(f"  {det_name:<25} {elapsed:.1f}s  raw={score:.3f} norm={norm:.3f}")
            return det_name, {"flag": bool(flag), "raw_score": score, "norm_score": round(norm, 3)}
        except Exception as e:
            print(f"  detector error [{det_name}]: {e}")
            return det_name, {"flag": False, "raw_score": -1.0, "norm_score": 0.0}

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_run_one, name, fn): name for name, fn in DETECTORS}
        for future in as_completed(futures):
            det_name, result = future.result()
            details[det_name] = result

    nb   = details.get("no_blink",    {}).get("norm_score", 0.0)
    sh   = details.get("static_head", {}).get("raw_score",  1.0)
    rppg = details.get("rppg_absence",{}).get("norm_score", 0.0)
    if nb >= 1.0 and sh < 0.04 and rppg > 0.40:
        details["photo_pattern"] = True
        print("  photo_pattern detected → FAKE")
        return False, "photo_pattern", details
    else:
        details["photo_pattern"] = False

    return True, "ok", details

def run_phase2(p1_details):
    if _lgbm_model is None:
        return None, "model_not_loaded", {}

    row = []
    for feat in LGBM_FEATURE_NAMES:
        det = p1_details.get(feat, {})
        score = det.get("raw_score", np.nan)
        if score == -1.0:
            score = np.nan
        row.append(score)

    import pandas as pd
    X = pd.DataFrame([row], columns=LGBM_FEATURE_NAMES)
    probs = _lgbm_model.predict_proba(X)[0]
    real_p, deepfake_p, physical_p = float(probs[0]), float(probs[1]), float(probs[2])

    print(f"  LightGBM probs: real={real_p:.3f} deepfake={deepfake_p:.3f} physical={physical_p:.3f}")

    if physical_p > LGBM_PHYSICAL_THRESHOLD:
        prediction = "FAKE"
        reason = f"lgbm_physical (p={physical_p:.3f})"
    elif real_p > LGBM_REAL_THRESHOLD:
        prediction = "REAL"
        reason = f"lgbm_real (p={real_p:.3f})"
    elif deepfake_p > LGBM_DEEPFAKE_THRESHOLD:
        prediction = "FAKE"
        reason = f"lgbm_deepfake (p={deepfake_p:.3f})"
    else:
        prediction = "UNCERTAIN → REAL"
        reason = f"lgbm_real (p={real_p:.3f})"

    lgbm_details = {
        "prob_real":     round(real_p, 4),
        "prob_deepfake": round(deepfake_p, 4),
        "prob_physical": round(physical_p, 4),
        "prediction":    prediction,
        "reason":        reason,
    }
    return prediction, reason, lgbm_details

def run_phase3(video_path):
    return predict_video_file(video_path, threshold=0.65)

def run_full_check(video_path):
    results = {}
    timings = {}
    t = time.time()
    p1_passed, p1_reason, p1_details = run_phase1(video_path)
    timings["phase1_ms"] = round((time.time() - t) * 1000)
    print(f"Phase 1 done in {timings['phase1_ms']}ms: {p1_reason}")

    results["phase1"] = "OK" if p1_passed else f"FAILED: {p1_reason}"
    results["phase1_details"] = p1_details

    if not p1_passed:
        results["timings"] = {**timings, "phase2_ms": None, "phase3_ms": None, "total_ms": timings["phase1_ms"]}
        results["deepfake"] = {"prediction": "FAKE", "reason": f"Phase1: {p1_reason}"}
        _log(video_path, results)
        return results

    t = time.time()
    prediction, reason, lgbm_details = run_phase2(p1_details)
    timings["phase2_ms"] = round((time.time() - t) * 1000)
    print(f"Phase 2 done in {timings['phase2_ms']}ms: {reason}")

    results["phase2"] = "OK" if prediction == "REAL" else f"FAILED: {reason}"
    results["phase2_details"] = lgbm_details

    if prediction == "FAKE":
        results["timings"] = {**timings, "phase3_ms": None, "total_ms": timings["phase1_ms"] + timings["phase2_ms"]}
        results["deepfake"] = {"prediction": "FAKE", "reason": reason}
        _log(video_path, results)
        return results

    t = time.time()
    deepfake_result = run_phase3(video_path)
    timings["phase3_ms"] = round((time.time() - t) * 1000)
    timings["total_ms"] = timings["phase1_ms"] + timings["phase2_ms"] + timings["phase3_ms"]
    print(f"Phase 3 done in {timings['phase3_ms']}ms: {deepfake_result}")

    results["phase3"] = "OK" if deepfake_result.get("prediction") == "REAL" else f"FAILED"
    results["timings"] = timings
    results["deepfake"] = deepfake_result
    _log(video_path, results)
    return results

def _log(video_path, results):
    print(f"\nResults for {video_path}:")
    for k, v in results.items():
        print(f"{k:<20}: {v}")