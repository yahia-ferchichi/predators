"""
Watch Random Agents - See fish with random behavior
====================================================
Run this to see the visualization with untrained (random) fish.
"""

import numpy as np
from environment import OceanEnvironment
from visualization import OceanRenderer, VideoRecorder
import time


def watch_random_agents(
    num_episodes: int = 5,
    record_video: bool = False,
    video_filename: str = "random_demo.mp4"
):
    """
    Watch fish with random actions trying to escape the shark.
    Great for seeing the initial (untrained) behavior.
    """
    print("\n" + "=" * 60)
    print("🐟 RANDOM AGENTS - Watch untrained fish!")
    print("=" * 60)
    print("Press ESC or close window to stop")
    print("=" * 60 + "\n")
    
    # Initialize environment and renderer
    env = OceanEnvironment(
        width=800,
        height=600,
        num_fish=15,
        num_rays=24,
        shark_speed=2.0,
        fish_speed=4.0
    )
    
    renderer = OceanRenderer(env.width, env.height)
    
    # Video recorder
    recorder = None
    if record_video:
        recorder = VideoRecorder(video_filename, env.width, env.height, fps=30)
    
    running = True
    total_survived = 0
    total_eaten = 0
    
    for episode in range(num_episodes):
        if not running:
            break
            
        print(f"\n🎬 Episode {episode + 1}/{num_episodes}")
        
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done and running:
            # Random actions for all fish
            actions = np.random.uniform(-1, 1, size=(2,))
            
            # Step environment
            obs, reward, done, truncated, info = env.step(actions)
            steps += 1
            
            # Move shark
            env._move_shark()
            
            # Apply random actions to each fish individually for variety
            for i in range(env.num_fish):
                if env.fish_alive[i]:
                    random_action = np.random.uniform(-1, 1, size=(2,))
                    env._move_fish(i, random_action)
            
            # Get render state and draw
            state = env.get_state_for_render()
            running = renderer.render(state, episode=episode + 1, training=False)
            
            # Capture frame for video
            if recorder:
                recorder.capture_frame(renderer.screen)
            
            # Check termination
            if np.sum(env.fish_alive) == 0:
                print(f"   💀 All fish eaten at step {steps}!")
                done = True
            
            if steps >= 500:  # Cap episode length for demo
                done = True
        
        # Episode stats
        alive = np.sum(env.fish_alive)
        eaten = env.num_fish - alive
        total_survived += alive
        total_eaten += eaten
        
        print(f"   ✅ Survived: {alive}/{env.num_fish} | Eaten: {eaten}")
        
        # Pause between episodes
        if running and episode < num_episodes - 1:
            time.sleep(0.5)
    
    # Save video
    if recorder:
        recorder.save()
    
    # Final stats
    print("\n" + "=" * 60)
    print("📊 FINAL STATS")
    print("=" * 60)
    print(f"Total fish survived: {total_survived}")
    print(f"Total fish eaten: {total_eaten}")
    print(f"Survival rate: {total_survived / (total_survived + total_eaten) * 100:.1f}%")
    print("=" * 60)
    
    renderer.close()
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    import sys
    
    record = "--record" in sys.argv
    watch_random_agents(num_episodes=3, record_video=record)
