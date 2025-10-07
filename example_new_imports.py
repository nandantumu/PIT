#!/usr/bin/env python3
"""
Example demonstrating the new import structure for pit.dynamics.singletrack

This example shows how to use the new unified import to access both
dynamics models and loss functions.
"""

import pit.dynamics.singletrack as st

def main():
    print("=" * 70)
    print("PIT Single Track - New Import Structure Example")
    print("=" * 70)
    
    # Show available exports
    print("\n1. Available Classes and Functions:")
    print("   Classes:")
    print(f"     - st.SingleTrack: {st.SingleTrack}")
    print(f"     - st.SingleTrackMod: {st.SingleTrackMod}")
    
    print("\n   Loss Functions:")
    print(f"     - st.yaw_normalized_loss: {st.yaw_normalized_loss}")
    print(f"     - st.yaw_normalized_loss_per_item: {st.yaw_normalized_loss_per_item}")
    print(f"     - st.yaw_normalized_loss_per_element: {st.yaw_normalized_loss_per_element}")
    
    print("\n   Constants:")
    print(f"     - st.ANGLE_INDICES: {st.ANGLE_INDICES}")
    print(f"     - st.non_angle_indices: {st.non_angle_indices}")
    
    # Example: Create a SingleTrack model
    print("\n2. Creating a SingleTrack model:")
    print("   model = st.SingleTrack(")
    print("       m=1225.887,")
    print("       Iz=1538.853,")
    print("       lf=0.88392,")
    print("       lr=1.50876,")
    print("       hcg=0.5,")
    print("       Csf=4.5,")
    print("       Csr=5.2,")
    print("       mu=0.9")
    print("   )")
    
    # Show usage with loss
    print("\n3. Using loss functions:")
    print("   # After integrating dynamics to get output_states")
    print("   loss = st.yaw_normalized_loss(output_states, target_states)")
    print("   loss_per_item = st.yaw_normalized_loss_per_item(output_states, target_states)")
    
    print("\n" + "=" * 70)
    print("✓ Import structure validated successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
