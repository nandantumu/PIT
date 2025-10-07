# Using the New Import Structure

The PIT library has been reorganized to allow importing single track dynamics and loss functions together under a unified namespace.

## New Import Style

```python
import pit.dynamics.singletrack as st

# Access dynamics classes
model = st.SingleTrack(m=1225, Iz=1538, lf=0.88, lr=1.51, hcg=0.5, Csf=4.5, Csr=5.2, mu=0.9)
model_mod = st.SingleTrackMod(m=1225, Iz=1538, lf=0.88, lr=1.51, hcg=0.5, Csf=4.5, Csr=5.2, mu=0.9)

# Access loss functions
loss = st.yaw_normalized_loss(output_states, target_states)
loss_per_item = st.yaw_normalized_loss_per_item(output_states, target_states)
loss_per_element = st.yaw_normalized_loss_per_element(output_states, target_states)

# Access constants
angle_indices = st.ANGLE_INDICES  # [4]
non_angle = st.non_angle_indices  # [0, 1, 2, 3, 5, 6]
```

## Available Exports

The `pit.dynamics.singletrack` module exports:

- **Dynamics Classes:**
  - `SingleTrack` - Standard single track vehicle dynamics model
  - `SingleTrackMod` - Modified single track vehicle dynamics model

- **Loss Functions:**
  - `yaw_normalized_loss` - Normalized loss with special handling for yaw angles
  - `yaw_normalized_loss_per_item` - Per-batch-item normalized loss
  - `yaw_normalized_loss_per_element` - Per-element normalized loss

- **Constants:**
  - `ANGLE_INDICES` - Indices of angular state variables
  - `non_angle_indices` - Indices of non-angular state variables

## Backward Compatibility

The old import paths still work:
```python
from pit.dynamics.single_track import SingleTrack
from pit.utilities.loss import yaw_normalized_loss
```

However, the new unified import is preferred for cleaner code organization.
