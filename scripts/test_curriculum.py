"""Test curriculum phase transitions at correct thresholds."""
from src.self_play import SelfPlayManager

def test_phase_transitions():
    """Verify phases switch at 1M, 2M, and 4M steps."""
    sp = SelfPlayManager()
    
    # Phase 1 tests
    assert sp.get_current_phase(0) == 1, "0 steps should be Phase 1"
    assert sp.get_current_phase(500_000) == 1, "500K steps should be Phase 1"
    assert sp.get_current_phase(999_999) == 1, "999,999 steps should be Phase 1"
    
    # Phase 2 tests
    assert sp.get_current_phase(1_000_000) == 2, "1M steps should be Phase 2"
    assert sp.get_current_phase(1_500_000) == 2, "1.5M steps should be Phase 2"
    assert sp.get_current_phase(1_999_999) == 2, "1,999,999 steps should be Phase 2"
    
    # Phase 3 tests
    assert sp.get_current_phase(2_000_000) == 3, "2M steps should be Phase 3"
    assert sp.get_current_phase(3_000_000) == 3, "3M steps should be Phase 3"
    assert sp.get_current_phase(3_999_999) == 3, "3,999,999 steps should be Phase 3"
    
    # Phase 4 tests
    assert sp.get_current_phase(4_000_000) == 4, "4M steps should be Phase 4"
    assert sp.get_current_phase(10_000_000) == 4, "10M steps should be Phase 4"
    
    print("✅ All phase transitions correct!")
    print("  - Phase 1: 0 - 999,999 steps")
    print("  - Phase 2: 1M - 1,999,999 steps")
    print("  - Phase 3: 2M - 3,999,999 steps")
    print("  - Phase 4: 4M+ steps")

if __name__ == "__main__":
    test_phase_transitions()
