def get_model_collection_from_file(path):
    return [l.strip().split(",") for l in open(path).readlines()]

model_collection = {
    "vswift": [
        ("vswift-g-14", "UnlabeledHybrid_5M_epoch20"),
    ],
    "vjepa": [
        ("vjepa-L-16", "VideoMix_2M_iterations90K"),
        ("vjepa-H-16", "VideoMix_2M_iterations90K"),
        ("vjepa-H-16-pix384", "VideoMix_2M_iterations90K"),
    ],
    "videomae_v1": [
        ("videomae-B-16", "K400_24W_epoch1600"),
        ("videomae-L-16", "K400_24W_epoch1600"),
        ("videomae-H-16", "K400_24W_epoch1600"),
        ("videomae-S-16", "SSV2_16W_epoch2400"),
        ("videomae-B-16", "SSV2_16W_epoch2400"),
    ],
    "videomae_v2": [
        ("videomae-g-14", "UnlabeledHybrid_1.34M_epoch1200"),
        ("videomae-g-14", "UnlabeledHybrid_1.34M_epoch1200_and_ppt_K710_65W"),
        ("videomae-S-16", "UnlabeledHybrid_1.34M_epoch1200_and_ppt_K710_65W_distillation"),
        ("videomae-B-16", "UnlabeledHybrid_1.34M_epoch1200_and_ppt_K710_65W_distillation"),
    ],
    "umt":[
        ("umt-B-16", "K710_65W_epoch200"),
        ("umt-L-16", "K710_65W_epoch200"),
        ("umt-B-16", "K710_65W_epoch200_and_ppt_K710_65W"),
        ("umt-L-16", "K710_65W_epoch200_and_ppt_K710_65W"),
    ],
    "internvideo_v2": [
        ("internvideov2s1-1B", "KMash_1.1M_epoch300"),
        ("internvideov2s1-S-14", "KMash_1.1M_epoch300_and_distillation"),
        ("internvideov2s1-B-14", "KMash_1.1M_epoch300_and_distillation"),
        ("internvideov2s1-L-14", "KMash_1.1M_epoch300_and_distillation"),
        ("internvideov2s1-1B", "KMash_1.1M_epoch300_and_ppt_K710_65W"),
        ("internvideov2s1-S-14", "KMash_1.1M_epoch300_and_ppt_K710_65W_distillation"),
        ("internvideov2s1-B-14", "KMash_1.1M_epoch300_and_ppt_K710_65W_distillation"),
        ("internvideov2s1-L-14", "KMash_1.1M_epoch300_and_ppt_K710_65W_distillation"),
    ],
    "viclip": [
        ("viclip-B-16", "InternVid_10MFLT"),
        ("viclip-B-16", "InternVid_200M"),
        ("viclip-L-14", "InternVid_10MFLT"),
    ],
    "internvideo_v1": [
        ("internvideov1-MM-L-14", "UnlabeledHybrid_14M"),
        ("internvideov1-videomae-B-16", "UnlabeledHybrid_1M"),
        ("internvideov1-videomae-L-16", "UnlabeledHybrid_1M"),
        ("internvideov1-videomae-H-16", "UnlabeledHybrid_1M"),
        ("internvideov1-videomae-B-16", "UnlabeledHybrid_1M_and_ppt_K710_65W"),
        ("internvideov1-videomae-L-16", "UnlabeledHybrid_1M_and_ppt_K700_65W"),
    ],
}
