class Args:
    # Dataset paths
    train_ir_dir = './MSRS-main/train/ir'
    train_vi_dir = './MSRS-main/train/vi'
    test_ir_dir = './MSRS-main/test/ir'
    test_vi_dir = './MSRS-main/test/vi'

    # Output paths
    save_model_dir = './model'
    save_loss_dir = './model/loss'
    output_dir = './outputs/DULRNet'
    model_path = './model/dulrnet_final.pth'
    vgg_model_path = './model/vgg/vgg16.pth'

    # Device and training settings
    cuda = True
    gpu_id = '0'
    seed = 2026
    epochs = 150
    batch_size = 8
    lr = 4e-4
    weight_decay = 1e-5
    warmup_epochs = 10
    eta_min = 1e-6
    save_interval = 10
    train_num = None

    # Image settings
    height = 128
    width = 128
    channel = 1

    # Network settings
    s = 3
    n = 128
    stride = 1
    num_stages = 8
    cd = 64

    # Loss weights reported in the paper
    lambda_pix = 0.20
    lambda_ssim = 0.24
    lambda_grad = 0.30
    lambda_feat = 0.39
    lambda_style = 0.15

    # Modality-specific loss settings
    gamma_ir = 4.0
    omega_vi = 0.10
    omega_ir = 0.90
    eta_vi = 0.05
    eta_ir = 0.95
    mu_grad = 3.0
    w_vi = 1.2
    w_ir_start = 2.5
    w_ir_end = 1.6

    # VGG feature-layer weights: relu1_2, relu2_2, relu3_3
    feat_weights = (0.1, 1.0 / 4.8, 1.0 / 3.7)
