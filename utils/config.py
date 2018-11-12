sample_rate = 16000
mu = 256

dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
                
residual_channels=32
dilation_channels=32
skip_channels=512
quantize_bins=256
global_condition_channels=32
global_condition_cardinality=377