# Fixed relation-support evaluation

Current report configuration: Market1501, 3368 query images, 19732 gallery
images, frozen Market-trained ResNet-50 or OSNet features evaluated separately.
Graph: k=20, gamma=0.4. Dynamic diffusion: alpha=5, beta=25, step=5,
depth=500 cumulative seeds. All queries and the complete gallery are evaluated.

Unified CSW+I-HyRDP uses rho=0.5 and ordinary endpoint-removed reciprocal-neighbour
Jaccard support, lam=0.25. Diagonal q is one; empty unions give zero. Weight:
W_ij * (rho if same camera else 1) * (1-lam+lam*q_ij). Weighted internal degrees,
seed boundaries and full-kernel bridge normalization are recomputed consistently.
Parameters and q remain fixed during a query; only seeds/local states change.

The historical development suite used 929 queries and 3376 gallery images per
feature model: 20 development configurations, followed by 8 full test runs and
4 mass-matched uniform controls. Development results were near saturation;
do not describe the backbone-seen development identities as unseen end-to-end
validation. The separate selected pure-camera rho=.25 control does not replace
the main model's pre-existing rho=.5. Ordinary/CB lam=.25 was frozen before test.

Ranking contains the dynamic gallery prefix followed by all remaining gallery
items in original feature-distance order. Same-ID same-camera matches and junk
are removed by the standard evaluator. Do not substitute 500 queries, a pilot
subset, numerical-tail fallback, or test-time tuning. Required residual <=1e-8;
MP refinement certifies the same float64 equation and actual selected nodes.

Changing feature files, source, graph, parameters or this protocol requires a new
output directory. Old archives retain their original protocol and source hashes;
this portable repository copy does not replace historical frozen evidence.

Reported mAP/Rank-1 (%): ResNet-50 85.8539935846/90.3503562945;
OSNet 93.2594329884/95.4869358670. These are historical verified measurements,
not a new full benchmark run performed by the code synchronization.
