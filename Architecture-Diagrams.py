from diagrams import Diagram, Node, Edge

with Diagram("MS_UNet Architecture", show=False):
    # Encoder: Downsampling path
    input_img     = Node("Input Image\n(3, 384, 512)")
    patch_embed   = Node("Patch Embedding")
    
    stage0        = Node("Stage 0\n(Transformer Blocks)")
    patch_merge1  = Node("Patch Merging")
    stage1        = Node("Stage 1\n(Transformer Blocks)")
    patch_merge2  = Node("Patch Merging")
    stage2        = Node("Stage 2\n(Transformer Blocks)")
    patch_merge3  = Node("Patch Merging")
    stage3        = Node("Stage 3\n(Transformer Blocks)")
    
    # Decoder: Upsampling path
    upsample4     = Node("Upsampling\n(Expanding Layer 4 + Fusion\nwith x2 from encoder)")
    stage4        = Node("Stage 4\n(Transformer Block)")
    upsample5     = Node("Upsampling\n(Expanding Layer 5 + Fusion\nwith x1 from encoder)")
    stage5        = Node("Stage 5\n(Transformer Block)")
    upsample6     = Node("Upsampling\n(Expanding Layer 6 + Fusion\nwith x0 from encoder)")
    stage6        = Node("Stage 6\n(Transformer Block)")
    
    regression    = Node("Regression Module")
    output        = Node("Output\n(8, 384, 512)")

    # Main vertical flow
    input_img >> patch_embed >> stage0 >> patch_merge1 >> stage1 >> patch_merge2 >> stage2 >> patch_merge3 >> stage3 >> upsample4 >> stage4 >> upsample5 >> stage5 >> upsample6 >> stage6 >> regression >> output

    # Skip connections (dashed edges)
    stage0 >> Edge(style="dashed", label="Skip") >> stage6
    stage1 >> Edge(style="dashed", label="Skip") >> stage5
    stage2 >> Edge(style="dashed", label="Skip") >> stage4
