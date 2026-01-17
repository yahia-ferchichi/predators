"""
Beautiful Ocean Visualization with Pygame
==========================================
Stunning visuals for the fish escape simulation.
"""

import pygame
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
import colorsys


class OceanRenderer:
    """
    High-quality renderer for the ocean environment.
    Features: water effects, bubbles, fish trails, shark animation, particles.
    """
    
    def __init__(self, width: int = 800, height: int = 600):
        pygame.init()
        pygame.display.set_caption("🦈 Deep Ocean - Fish vs Shark")
        
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        
        # Colors - Deep ocean theme
        self.colors = {
            "deep_blue": (8, 24, 58),
            "mid_blue": (15, 42, 82),
            "light_blue": (25, 62, 110),
            "water_highlight": (40, 90, 140),
            "shark_body": (70, 80, 95),
            "shark_belly": (140, 150, 160),
            "shark_eye": (20, 20, 25),
            "shark_teeth": (240, 240, 245),
            "fish_orange": (255, 140, 50),
            "fish_yellow": (255, 200, 60),
            "fish_blue": (80, 180, 255),
            "fish_green": (100, 230, 150),
            "fish_pink": (255, 130, 180),
            "bubble": (180, 220, 255),
            "particle": (100, 200, 255),
            "text": (200, 220, 240),
        }
        
        # Fish color palette (for variety)
        self.fish_colors = [
            ((255, 140, 50), (255, 180, 100)),    # Orange
            ((255, 200, 60), (255, 230, 120)),    # Yellow
            ((80, 180, 255), (150, 210, 255)),    # Blue
            ((100, 230, 150), (170, 255, 200)),   # Green
            ((255, 130, 180), (255, 180, 210)),   # Pink
            ((200, 100, 255), (230, 160, 255)),   # Purple
        ]
        
        # Particles (bubbles, debris)
        self.bubbles: List[Dict] = []
        self.particles: List[Dict] = []
        self.fish_trails: Dict[int, List[Tuple[float, float]]] = {}
        
        # Animation state
        self.time = 0
        self.water_offset = 0
        
        # Font
        pygame.font.init()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # Pre-generate water gradient
        self._create_water_gradient()
        
        # Initialize bubbles
        for _ in range(30):
            self._spawn_bubble()
    
    def _create_water_gradient(self):
        """Create a vertical gradient for the water background."""
        self.water_surface = pygame.Surface((self.width, self.height))
        for y in range(self.height):
            t = y / self.height
            # Gradient from deep to lighter blue (top to bottom inverted for underwater feel)
            r = int(self.colors["deep_blue"][0] + (self.colors["light_blue"][0] - self.colors["deep_blue"][0]) * (1 - t * 0.7))
            g = int(self.colors["deep_blue"][1] + (self.colors["light_blue"][1] - self.colors["deep_blue"][1]) * (1 - t * 0.7))
            b = int(self.colors["deep_blue"][2] + (self.colors["light_blue"][2] - self.colors["deep_blue"][2]) * (1 - t * 0.7))
            pygame.draw.line(self.water_surface, (r, g, b), (0, y), (self.width, y))
    
    def _spawn_bubble(self, x: Optional[float] = None, y: Optional[float] = None):
        """Spawn a bubble particle."""
        bubble = {
            "x": x if x else np.random.uniform(0, self.width),
            "y": y if y else np.random.uniform(0, self.height),
            "size": np.random.uniform(2, 8),
            "speed": np.random.uniform(0.5, 2),
            "wobble": np.random.uniform(0, 2 * np.pi),
            "wobble_speed": np.random.uniform(0.02, 0.08),
        }
        self.bubbles.append(bubble)
    
    def _spawn_particle(self, x: float, y: float, color: Tuple[int, int, int]):
        """Spawn a particle effect."""
        for _ in range(3):
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(1, 3)
            self.particles.append({
                "x": x,
                "y": y,
                "vx": np.cos(angle) * speed,
                "vy": np.sin(angle) * speed,
                "life": 1.0,
                "decay": np.random.uniform(0.02, 0.05),
                "color": color,
                "size": np.random.uniform(2, 5),
            })
    
    def _update_effects(self):
        """Update all particle effects."""
        self.time += 1
        self.water_offset = math.sin(self.time * 0.02) * 3
        
        # Update bubbles
        for bubble in self.bubbles[:]:
            bubble["y"] -= bubble["speed"]
            bubble["wobble"] += bubble["wobble_speed"]
            bubble["x"] += math.sin(bubble["wobble"]) * 0.5
            
            if bubble["y"] < -20:
                self.bubbles.remove(bubble)
                self._spawn_bubble(y=self.height + 10)
        
        # Update particles
        for particle in self.particles[:]:
            particle["x"] += particle["vx"]
            particle["y"] += particle["vy"]
            particle["vy"] += 0.05  # Gravity
            particle["life"] -= particle["decay"]
            
            if particle["life"] <= 0:
                self.particles.remove(particle)
    
    def _draw_water_caustics(self):
        """Draw animated light caustics on the water."""
        caustic_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        for i in range(5):
            x = (self.time * 0.5 + i * 200) % (self.width + 200) - 100
            for j in range(3):
                y = j * 200 + math.sin(self.time * 0.03 + i) * 30
                alpha = int(15 + 10 * math.sin(self.time * 0.05 + i + j))
                size = 150 + 50 * math.sin(self.time * 0.02 + i * 0.5)
                pygame.draw.ellipse(
                    caustic_surface,
                    (*self.colors["water_highlight"], alpha),
                    (x - size/2, y - size/4, size, size/2)
                )
        
        self.screen.blit(caustic_surface, (0, 0))
    
    def _draw_bubbles(self):
        """Draw bubble particles."""
        for bubble in self.bubbles:
            x = int(bubble["x"])
            y = int(bubble["y"])
            size = int(bubble["size"])
            
            # Bubble with highlight
            pygame.draw.circle(self.screen, (*self.colors["bubble"], 100), (x, y), size)
            pygame.draw.circle(self.screen, (255, 255, 255), (x - size//3, y - size//3), max(1, size//4))
    
    def _draw_particles(self):
        """Draw particle effects."""
        for particle in self.particles:
            alpha = int(particle["life"] * 255)
            color = (*particle["color"][:3], alpha)
            size = int(particle["size"] * particle["life"])
            
            surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, color, (size, size), size)
            self.screen.blit(surf, (int(particle["x"] - size), int(particle["y"] - size)))
    
    def _draw_shark(self, pos: np.ndarray, direction: int, size: Tuple[float, float]):
        """Draw the shark with animation."""
        x, y = pos
        w, h = size
        
        # Flip based on direction
        flip = direction < 0
        
        # Body wobble animation
        wobble = math.sin(self.time * 0.15) * 3
        
        # Main body (ellipse)
        body_color = self.colors["shark_body"]
        belly_color = self.colors["shark_belly"]
        
        # Create shark surface for potential flipping
        shark_surf = pygame.Surface((int(w * 1.5), int(h * 2)), pygame.SRCALPHA)
        shark_w, shark_h = shark_surf.get_size()
        cx, cy = shark_w // 2, shark_h // 2
        
        # Tail
        tail_points = [
            (cx - w * 0.4, cy),
            (cx - w * 0.7, cy - h * 0.5 + wobble),
            (cx - w * 0.5, cy),
            (cx - w * 0.7, cy + h * 0.4 - wobble),
        ]
        pygame.draw.polygon(shark_surf, body_color, tail_points)
        
        # Main body ellipse
        pygame.draw.ellipse(shark_surf, body_color, 
                          (cx - w * 0.4, cy - h * 0.35, w * 0.9, h * 0.7))
        
        # Belly (lighter area)
        pygame.draw.ellipse(shark_surf, belly_color,
                          (cx - w * 0.3, cy, w * 0.6, h * 0.3))
        
        # Dorsal fin
        fin_points = [
            (cx, cy - h * 0.35),
            (cx - w * 0.1, cy - h * 0.6 + wobble * 0.5),
            (cx + w * 0.15, cy - h * 0.35),
        ]
        pygame.draw.polygon(shark_surf, body_color, fin_points)
        
        # Pectoral fins
        pfin_points = [
            (cx + w * 0.1, cy + h * 0.1),
            (cx + w * 0.3, cy + h * 0.4 - wobble * 0.3),
            (cx + w * 0.2, cy + h * 0.15),
        ]
        pygame.draw.polygon(shark_surf, body_color, pfin_points)
        
        # Head / snout
        pygame.draw.ellipse(shark_surf, body_color,
                          (cx + w * 0.15, cy - h * 0.25, w * 0.4, h * 0.5))
        
        # Eye
        eye_x = cx + w * 0.25
        eye_y = cy - h * 0.1
        pygame.draw.circle(shark_surf, self.colors["shark_eye"], (int(eye_x), int(eye_y)), 6)
        pygame.draw.circle(shark_surf, (60, 60, 70), (int(eye_x), int(eye_y)), 3)
        
        # Gills
        for i in range(3):
            gx = cx + w * 0.05 - i * 8
            pygame.draw.line(shark_surf, (50, 60, 75), 
                           (gx, cy - h * 0.15), (gx, cy + h * 0.1), 2)
        
        # Teeth (menacing!)
        mouth_y = cy + h * 0.05
        for i in range(5):
            tx = cx + w * 0.35 + i * 6
            pygame.draw.polygon(shark_surf, self.colors["shark_teeth"], [
                (tx, mouth_y),
                (tx + 3, mouth_y + 8),
                (tx + 6, mouth_y),
            ])
        
        # Flip if needed
        if flip:
            shark_surf = pygame.transform.flip(shark_surf, True, False)
        
        # Draw to screen
        self.screen.blit(shark_surf, (int(x - shark_w // 2), int(y - shark_h // 2)))
        
        # Spawn occasional bubbles from shark
        if np.random.random() < 0.1:
            self._spawn_bubble(x - direction * w * 0.3, y)
    
    def _draw_fish(self, idx: int, pos: np.ndarray, vel: np.ndarray, 
                   radius: float, alive: bool):
        """Draw a fish with trail effect."""
        if not alive:
            # Death particles
            return
        
        x, y = pos
        vx, vy = vel
        
        # Get fish color based on index
        base_color, highlight_color = self.fish_colors[idx % len(self.fish_colors)]
        
        # Direction fish is facing
        speed = np.linalg.norm(vel)
        if speed > 0.1:
            angle = math.atan2(vy, vx)
        else:
            angle = 0
        
        # Update trail
        if idx not in self.fish_trails:
            self.fish_trails[idx] = []
        
        trail = self.fish_trails[idx]
        trail.append((x, y))
        if len(trail) > 15:
            trail.pop(0)
        
        # Draw trail
        for i, (tx, ty) in enumerate(trail):
            alpha = int((i / len(trail)) * 100)
            t_size = int(radius * 0.5 * (i / len(trail)))
            if t_size > 0:
                trail_surf = pygame.Surface((t_size * 2, t_size * 2), pygame.SRCALPHA)
                pygame.draw.circle(trail_surf, (*base_color, alpha), (t_size, t_size), t_size)
                self.screen.blit(trail_surf, (int(tx - t_size), int(ty - t_size)))
        
        # Fish body
        fish_length = radius * 2.5
        fish_height = radius * 1.5
        
        # Create rotated fish
        fish_surf = pygame.Surface((int(fish_length * 2), int(fish_height * 2)), pygame.SRCALPHA)
        cx, cy = fish_surf.get_width() // 2, fish_surf.get_height() // 2
        
        # Tail wobble
        wobble = math.sin(self.time * 0.3 + idx) * 5
        
        # Tail
        tail_points = [
            (cx - fish_length * 0.3, cy),
            (cx - fish_length * 0.6, cy - fish_height * 0.4 + wobble),
            (cx - fish_length * 0.4, cy),
            (cx - fish_length * 0.6, cy + fish_height * 0.4 - wobble),
        ]
        pygame.draw.polygon(fish_surf, base_color, tail_points)
        
        # Body
        pygame.draw.ellipse(fish_surf, base_color,
                          (cx - fish_length * 0.3, cy - fish_height * 0.35,
                           fish_length * 0.7, fish_height * 0.7))
        
        # Highlight/stripe
        pygame.draw.ellipse(fish_surf, highlight_color,
                          (cx - fish_length * 0.1, cy - fish_height * 0.2,
                           fish_length * 0.4, fish_height * 0.15))
        
        # Eye
        pygame.draw.circle(fish_surf, (30, 30, 40), (int(cx + fish_length * 0.15), int(cy - 2)), 4)
        pygame.draw.circle(fish_surf, (255, 255, 255), (int(cx + fish_length * 0.15), int(cy - 2)), 2)
        
        # Rotate fish to face movement direction
        angle_deg = math.degrees(angle)
        rotated = pygame.transform.rotate(fish_surf, -angle_deg)
        
        # Draw
        rect = rotated.get_rect(center=(int(x), int(y)))
        self.screen.blit(rotated, rect)
    
    def _draw_ui(self, state: Dict, fps: float, episode: int = 0, training: bool = False):
        """Draw UI overlay."""
        alive_count = np.sum(state["fish_alive"])
        total_fish = len(state["fish_alive"])
        
        # Semi-transparent panel
        panel = pygame.Surface((200, 120), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 150))
        self.screen.blit(panel, (10, 10))
        
        # Stats
        title = "TRAINING" if training else "SIMULATION"
        title_surf = self.font_medium.render(title, True, self.colors["text"])
        self.screen.blit(title_surf, (20, 15))
        
        if episode > 0:
            ep_surf = self.font_small.render(f"Episode: {episode}", True, self.colors["text"])
            self.screen.blit(ep_surf, (20, 45))
        
        alive_surf = self.font_small.render(f"Fish: {alive_count}/{total_fish}", True, 
                                           (100, 255, 150) if alive_count > total_fish // 2 else (255, 100, 100))
        self.screen.blit(alive_surf, (20, 70))
        
        fps_surf = self.font_small.render(f"FPS: {int(fps)}", True, self.colors["text"])
        self.screen.blit(fps_surf, (20, 95))
        
        # Step counter (bottom right)
        step_surf = self.font_small.render(f"Step: {state['steps']}", True, self.colors["text"])
        self.screen.blit(step_surf, (self.width - 100, self.height - 30))
    
    def render(self, state: Dict, episode: int = 0, training: bool = False) -> bool:
        """
        Render the current state.
        Returns False if window was closed.
        """
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        
        # Update effects
        self._update_effects()
        
        # Draw background
        self.screen.blit(self.water_surface, (0, 0))
        self._draw_water_caustics()
        
        # Draw bubbles (behind everything)
        self._draw_bubbles()
        
        # Draw shark
        self._draw_shark(
            state["shark_pos"],
            state["shark_direction"],
            state["shark_size"]
        )
        
        # Draw fish
        for i in range(len(state["fish_positions"])):
            self._draw_fish(
                i,
                state["fish_positions"][i],
                state["fish_velocities"][i],
                state["fish_radius"],
                state["fish_alive"][i]
            )
            
            # Spawn death particles if fish just died
            if not state["fish_alive"][i] and i in self.fish_trails:
                pos = state["fish_positions"][i]
                color = self.fish_colors[i % len(self.fish_colors)][0]
                self._spawn_particle(pos[0], pos[1], color)
                del self.fish_trails[i]
        
        # Draw particles (on top)
        self._draw_particles()
        
        # Draw UI
        fps = self.clock.get_fps()
        self._draw_ui(state, fps, episode, training)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(60)
        
        return True
    
    def close(self):
        """Clean up pygame."""
        pygame.quit()


class VideoRecorder:
    """Record frames to video file."""
    
    def __init__(self, filename: str, width: int, height: int, fps: int = 30):
        self.filename = filename
        self.width = width
        self.height = height
        self.fps = fps
        self.frames: List[np.ndarray] = []
    
    def capture_frame(self, screen: pygame.Surface):
        """Capture current pygame screen."""
        frame = pygame.surfarray.array3d(screen)
        frame = np.transpose(frame, (1, 0, 2))  # Pygame uses (width, height)
        self.frames.append(frame)
    
    def save(self):
        """Save captured frames to video."""
        if not self.frames:
            print("No frames to save!")
            return
        
        try:
            import cv2
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.filename, fourcc, self.fps, (self.width, self.height))
            
            for frame in self.frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            print(f"✅ Video saved: {self.filename}")
            
        except ImportError:
            print("OpenCV not available, saving as GIF...")
            try:
                from PIL import Image
                
                images = [Image.fromarray(frame) for frame in self.frames[::2]]  # Skip frames for smaller GIF
                gif_name = self.filename.replace('.mp4', '.gif')
                images[0].save(gif_name, save_all=True, append_images=images[1:], 
                             duration=1000//self.fps * 2, loop=0)
                print(f"✅ GIF saved: {gif_name}")
            except Exception as e:
                print(f"❌ Could not save video: {e}")
