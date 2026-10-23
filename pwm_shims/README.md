# pwm_shims

`pytorch3d` is a symlink into the pwm211 env's site-packages. It exists so the
PWM loader can be imported inside `pointworld-env`, which has no pytorch3d of
its own.

Only `pytorch3d.transforms.matrix_to_quaternion` and `quaternion_to_axis_angle`
are ever reached — pure tensor code with no compiled extension — and they are
reached from `utils/pose_ops.so3_log`, on the yikai pose channels that
PointWorld discards. `pwm_stream.py` appends this directory to the END of
`sys.path` (env `PWM_PYTORCH3D`), so the local torch and every local package
keep priority.

Delete the symlink and unset `PWM_PYTORCH3D` once pointworld-env has its own
pytorch3d.
