# Range-of-motion tables for hand pose rectification

Siformer's rectification code (`external/Siformer/rectifications/`) needs two table files that the
authors' repository does not publish. This project rebuilt them. The angle ranges come from Table 1 of
the paper; the paper does not say between which three points each angle is measured, so that part is
the project's own choice, following MediaPipe's 21 hand points (0 wrist, 1–4 thumb, 5–8 index,
9–12 middle, 13–16 ring, 17–20 little).

Two sets:

- `full/`: all rows described below (4 abduction–adduction rows, 15 flexion–extension rows).
- `dip_only/` (default): only the 4 DIP flexion–extension rows of the long fingers.

## Why `dip_only`

On the MediaPipe training set (35,640 hands), `full` corrects at least one joint in **78%** of the
hands; on the authors' data the rate is similar (85%). Abduction–adduction rows fire on 26–32% of the
hands, thumb MCP on 38%, long-finger MCP and PIP on 7–16%. The ranges in Table 1 are limits of real
joints in 3D, while the angles here are measured on a 2D projection: a fist or a finger pointing at the
camera exceeds the limit in the image although the pose is normal. Only the DIP rows fire on fewer
than 3% of the hands, consistent with MediaPipe errors being rare.

The data version was chosen on the validation set (none, `dip_only`, `full`): `dip_only` was best by
about one point. Over three seeds on the test set, rectification brings no measurable gain
(64.3% against 64.1% without it).

## fe_active_motion.csv: flexion–extension

`paired_joint_index = [refer, check, extra]`. In the authors' code the angle is between the segments
`extra → check` and `check → refer`, i.e. the flexion at joint `check`. If it exceeds the range, the
segment `check → refer` is rotated back by a fraction α (0.4), and the points in
`associated_joint_index` move with it.

| Joint | Points (refer, check, extra) | Range in the paper | Range in the table |
|---|---|---|---|
| Thumb CMC | 2, 1, 0 | −20 … 45 | [−45, 20] |
| Thumb MCP | 3, 2, 1 | 0 … 80 | [−80, 0] |
| Thumb IP | 4, 3, 2 | −30 … 90 | [−90, 30] |
| MCP, long fingers | b+1, b, 0 | −40 … 90 | [−40, 90] |
| PIP | b+2, b+1, b | 0 … 130 | [0, 130] |
| DIP | b+3, b+2, b+1 | −30 … 90 | [−30, 90] |

(b = 5, 9, 13, 17.)

- **Thumb:** the authors' code uses signed ranges. With the y-axis pointing up and a "clockwise"
  flexion direction, thumb flexion gives a negative angle, so the ranges are sign-flipped. Checked on
  synthetic hands, palm and back facing the camera: natural flexion is left alone, hyperextension is
  corrected (`tests/selftest.py`).
- **Long fingers:** the code only uses `[-max, max]`, so the sign does not matter. On a 2D image this
  angle only catches excessive flexion.

## aa_active_motion.csv: abduction–adduction

`paired_joint_index = [check, refer, centre]`: the angle between `centre → check` and
`centre → refer`.

- Only the MCP of the four long fingers, ±15° as in the paper, with check = PIP, refer = MCP,
  centre = wrist: the sideways deviation of the proximal phalanx from the metacarpal.
- **Thumb abduction–adduction is not used:** on a 2D image its sign flips when the back of the hand
  faces the camera, so it would correct the wrong hands.

## Before using new tables

Run `python src/rectify_hands.py --inp data/csv/train.csv --tables <set> --stats <file>.json` and look
at the share of hands each row corrects. MediaPipe errors are rare, so a row that fires on many hands
is measuring the wrong thing and should be removed or fixed.
