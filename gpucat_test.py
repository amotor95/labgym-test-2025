from analyzebehavior_gpu import AnalyzeAnimalGPU

path_to_detector = "/Users/jackmcclure/Desktop/pip_LabGym_version_control/gpucat_testing/MouseBoxes_TopView_960"

path_to_clip = "/Users/jackmcclure/Desktop/pip_LabGym_version_control/gpucat_testing/mouse_grooming_clip.mp4"

path_to_

analyzer = AnalyzeAnimalGPU(path_to_detector, path_to_clip, 300)

analyzer.pre_process()