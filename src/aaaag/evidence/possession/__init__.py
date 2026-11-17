"""Team-in-possession evidence (lane: possession) -- given the ball is alive, which team has it?

Producers here mostly speak to the H/A split: ball proximity and control, touches from ball
kinematics, team shape and movement while the ball is missing, Smart Tagging phase priors, learned
frame classifiers trained on OTHER matches. One module per idea; register each in EVIDENCE.
"""
