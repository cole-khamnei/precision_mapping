import os
import sys

# ----------------------------------------------------------------------------# 
# --------------------               Paths                --------------------# 
# ----------------------------------------------------------------------------# 

PM_SRC_PATH = os.path.dirname(os.path.abspath(__file__))
PRECISION_MAPPING_DIR = os.path.realpath(f"{PM_SRC_PATH}/../")
RESOURCES_DIR = f"{PRECISION_MAPPING_DIR}/resources"

EXAMPLE_DTSERIES = os.path.join(PRECISION_MAPPING_DIR, "tests", "example.dtseries.nii")

# ----------------------------------------------------------------------------# 
# --------------------           Resource Paths           --------------------# 
# ----------------------------------------------------------------------------# 

NETWORK_PRIORS = os.path.join(PRECISION_MAPPING_DIR, "resources", "network_FC_and_spatial_priors.mat")

NETWORK_CMAP = {
    'Auditory': [170, 83, 246, 255],
    'CinguloOpercular/Action-mode': [70, 10, 146, 255],
    'Default_Anterolateral': [145, 24, 103, 255],
    'Default_Dorsolateral': [213, 156, 65, 255],
    'Default_Parietal': [230, 53, 35, 255],
    'Default_Retrosplenial': [255, 254, 217, 255],
    'DorsalAttention': [96, 213, 60, 255],
    'Frontoparietal': [254, 253, 84, 255],
    'MedialParietal': [2, 87, 255, 255],
    'None': [198, 198, 198, 255],
    'Premotor/DorsalAttentionII': [253, 130, 255, 255],
    'Salience': [0, 0, 0, 255],
    'SomatoCognitiveAction': [120, 19, 20, 255],
    'Somatomotor_Face': [254, 126, 0, 255],
    'Somatomotor_Foot': [12, 80, 4, 255],
    'Somatomotor_Hand': [113, 253, 254, 255],
    'Visual_Dorsal/VentralStream': [47, 114, 180, 255],
    'Visual_Lateral': [27, 14, 145, 255],
    'Visual_V1': [181, 209, 140, 255],
    'Visual_V5': [254, 180, 97, 255],
    'Language': [62, 153, 153, 255]
}
NETWORK_CMAP = {k: [v_i/255 for v_i in v] for k,v in NETWORK_CMAP.items()}

# ----------------------------------------------------------------------------# 
# --------------------                End                 --------------------# 
# ----------------------------------------------------------------------------#
