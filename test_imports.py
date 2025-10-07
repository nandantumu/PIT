#!/usr/bin/env python3
"""
Test script to verify the new import structure works correctly.
This demonstrates the desired import notation: import pit.dynamics.singletrack as st
"""

# Test importing singletrack as st
import pit.dynamics.singletrack as st

# Test that we can access the dynamics classes
print("SingleTrack class:", st.SingleTrack)
print("SingleTrackMod class:", st.SingleTrackMod)

# Test that we can access the loss functions
print("yaw_normalized_loss function:", st.yaw_normalized_loss)
print("yaw_normalized_loss_per_item function:", st.yaw_normalized_loss_per_item)
print("yaw_normalized_loss_per_element function:", st.yaw_normalized_loss_per_element)

# Test that we can access constants
print("ANGLE_INDICES:", st.ANGLE_INDICES)
print("non_angle_indices:", st.non_angle_indices)

print("\n✓ All imports successful! The new structure works as expected.")
