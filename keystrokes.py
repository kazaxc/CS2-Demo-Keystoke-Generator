from demoparser2 import DemoParser
import numpy as np
import cv2, argparse, sys, os, configparser
from pathlib import Path
from typing import Dict, List, Tuple, Any

event_names = [
    "begin_new_match", "round_start", "round_end", "round_mvp", 
    "player_death", "bomb_planted", "bomb_defused", "hostage_rescued", 
    "weapon_fire", "flashbang_detonate", "hegrenade_detonate", 
    "molotov_detonate", "smokegrenade_detonate", "player_hurt", 
    "player_blind"
]

KEY_MAPPING = {
    "IN_ATTACK": 1 << 0,
    "IN_JUMP": 1 << 1,
    "IN_DUCK": 1 << 2,
    "IN_FORWARD": 1 << 3,
    "IN_BACK": 1 << 4,
    "IN_USE": 1 << 5, 
    "IN_CANCEL": 1 << 6,
    "IN_TURNLEFT": 1 << 7,
    "IN_TURNRIGHT": 1 << 8,
    "IN_MOVELEFT": 1 << 9,
    "IN_MOVERIGHT": 1 << 10,
    "IN_ATTACK2": 1 << 11,
    "IN_RELOAD": 1 << 13,
    "IN_ALT1": 1 << 14,
    "IN_ALT2": 1 << 15,
    "IN_SPEED": 1 << 16,
    "IN_WALK": 1 << 17,
    "IN_ZOOM": 1 << 18,
    "IN_WEAPON1": 1 << 19,
    "IN_WEAPON2": 1 << 20,
    "IN_BULLRUSH": 1 << 21,
    "IN_GRENADE1": 1 << 22,
    "IN_GRENADE2": 1 << 23,
    "IN_ATTACK3": 1 << 24,
    "UNKNOWN_25": 1 << 25,
    "UNKNOWN_26": 1 << 26,
    "UNKNOWN_27": 1 << 27,
    "UNKNOWN_28": 1 << 28,
    "UNKNOWN_29": 1 << 29,
    "UNKNOWN_30": 1 << 30,
    "UNKNOWN_31": 1 << 31,
    "IN_SCORE": 1 << 33,
    "IN_INSPECT": 1 << 35,
}

KEY_DISPLAY = {
    "IN_FORWARD": "W",
    "IN_BACK": "S",
    "IN_MOVELEFT": "A",
    "IN_MOVERIGHT": "D",
    "IN_ATTACK": "M1",
    "IN_ATTACK2": "M2",
    "IN_JUMP": "Space",
    "IN_DUCK": "Ctrl",
    "IN_WALK": "Shift",
}

def extract_buttons(buttons: int):
    all_keys_str = []
    for (button_name, bit_slot) in KEY_MAPPING.items():
        if (buttons & bit_slot) != 0:
            all_keys_str.append(button_name)

    return all_keys_str

# Keyboard visualization
def create_keyboard_overlay(pressed_keys: List[str], player_name: str, player_color: Tuple[int, int, int]) -> np.ndarray:
    """Create a keyboard visualization overlay"""
    # Create keyboard image with green background for chroma keying
    kb_width, kb_height = 600, 400
    keyboard_img = np.zeros((kb_height, kb_width, 3), dtype=np.uint8)
    keyboard_img[:] = (0, 255, 0)  # Set background to green

    # Key positions (x, y, width, height) - scaled up
    key_positions = {
        'W': (280, 80, 60, 50),
        'A': (200, 160, 60, 50),
        'S': (280, 160, 60, 50),
        'D': (360, 160, 60, 50),
        'SPACE': (160, 280, 240, 40),
        'CTRL': (40, 280, 80, 40),
        'SHIFT': (40, 230, 100, 40),
        'M1': (440, 80, 60, 50),
        'M2': (520, 80, 60, 50),
    }
    
    # Draw keys
    for key_name, (x, y, w, h) in key_positions.items():
        # Check if this key is pressed
        is_pressed = any(KEY_DISPLAY.get(pressed_key, pressed_key) == key_name for pressed_key in pressed_keys)
        
        # Key color and text color these should be green for chroma keying then red for when pressed with white text
        if is_pressed:
            color = (0, 0, 255)  # Red for pressed keys
            text_color = (255, 255, 255)
        else:
            color = (0, 255, 0)  # Green for unpressed keys
            text_color = (255, 255, 255)

        # Draw key background colour is green for chroma keying
        cv2.rectangle(keyboard_img, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(keyboard_img, (x, y), (x + w, y + h), (255, 255, 255), 2)


        # Draw key text
        font_scale = 0.8 if len(key_name) <= 2 else 0.6
        text_size = cv2.getTextSize(key_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(keyboard_img, key_name, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2)
    
    # Draw mouse shape instead of keys - TODO
    
    return keyboard_img

def process_demo(demo_path: Path, output_dir: Path, player_name_str: str = None):
    """Process the demo file and create videos for each player or a specific player if provided"""

    parser = DemoParser(str(demo_path))

    # Parse all events
    all_events = parser.parse_events(event_names, other=["game_time", "team_num"])

    if not all_events:
        print("No events found in the demo.")
        return
    
    # Find match start tick
    begin_new_match_df = next((df for event_name, df in all_events if event_name == 'begin_new_match'), None)
    match_start_tick = begin_new_match_df['tick'].iloc[0] if begin_new_match_df is not None else 0

    # Filter out events before the match start
    filtered_events = [(event_name, df[df['tick'] >= match_start_tick]) for event_name, df in all_events]

    wanted_props = ["player_name", "buttons"]
    tick_values = set()
    for _, df in filtered_events:
        tick_values.update(df['tick'].unique())
    all_ticks = parser.parse_ticks(wanted_props, ticks=list(tick_values))


    # Convert ticks to a map
    all_ticks_map = {}
    for tick in all_ticks.itertuples():
        if tick.tick not in all_ticks_map:
            all_ticks_map[tick.tick] = []
        all_ticks_map[tick.tick].append(tick)

    # Access ticks
    game_end_events = next((df for event_name, df in filtered_events if event_name == 'round_end'), None)
    game_end_tick = max(game_end_events['tick']) if game_end_events is not None else 0
    scoreboard = all_ticks_map.get(game_end_tick, [])

    # Get all players' names
    all_player_names = {tick.player_name for tick in all_ticks_map.get(match_start_tick, [])}
    print("All players in the match:")
    for player_name in all_player_names:
        print(player_name)

    # Get inputs for each player throughout the game
    keystrokes_per_player = {}

    if scoreboard:
        for tick in all_ticks_map:
            for player in all_ticks_map[tick]:
                if player.player_name not in all_player_names:
                    continue
                if player.player_name not in keystrokes_per_player:
                    keystrokes_per_player[player.player_name] = []
                keystrokes_per_player[player.player_name].append((tick, player.buttons))

    if player_name_str:
        create_video_for_player(player_name_str, keystrokes_per_player.get(player_name_str, []), f"{output_dir}/{player_name_str}_keystrokes.mp4")
    else:
        for player_name, keystrokes in keystrokes_per_player.items():
            create_video_for_player(player_name, keystrokes, f"{output_dir}/{player_name}_keystrokes.mp4")



def create_video_for_player(player_name: str, keystrokes: List[Tuple[int, int]], output_path: str):
    """Create a video for a single player showing their keystrokes at 60fps (game tickrate 64/s)"""
    # Load configuration
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    video_fps = config.getint('SETTINGS', 'fps', fallback=60)
    video_width = config.getint('SETTINGS', 'width', fallback=800)
    video_height = config.getint('SETTINGS', 'height', fallback=600)
    tickrate = config.getint('SETTINGS', 'tickrate', fallback=64)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_fps, (video_width, video_height))

    player_color = (0, 255, 0)  # Green color for the player

    # Sort keystrokes by tick
    keystrokes = sorted(keystrokes, key=lambda x: x[0])
    if not keystrokes:
        print(f"No keystrokes for {player_name}")
        out.release()
        return

    # Build a list of (tick, buttons) and fill gaps with previous value
    tick_to_buttons = {}
    for tick, buttons in keystrokes:
        tick_to_buttons[tick] = buttons

    min_tick = keystrokes[0][0]
    max_tick = keystrokes[-1][0]
    total_game_seconds = (max_tick - min_tick) / tickrate
    total_frames = int(total_game_seconds * video_fps) + 1

    # Prepare for fast lookup
    sorted_ticks = sorted(tick_to_buttons.keys())
    current_idx = 0

    for frame_idx in range(total_frames):
        # Calculate the corresponding game time and tick for this frame
        game_time = frame_idx / video_fps
        tick_float = min_tick + game_time * tickrate
        # Find the latest tick <= tick_float
        while (current_idx + 1 < len(sorted_ticks)) and (sorted_ticks[current_idx + 1] <= tick_float):
            current_idx += 1
        tick = sorted_ticks[current_idx]
        buttons = tick_to_buttons[tick]
        if np.isnan(buttons):
            buttons = 0
        pressed_button = int(buttons)
        pressed_keys = extract_buttons(pressed_button)
        frame = create_keyboard_overlay(pressed_keys, player_name, player_color)
        frame_resized = cv2.resize(frame, (video_width, video_height))
        out.write(frame_resized)

    out.release()
    print(f"Video for {player_name} saved to {output_path}")

def main() -> None:
    """Main function to run the keystroke visualization"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create keystroke videos for players in a CS2 demo.")
    parser.add_argument("--demo", type=Path, required=True, help="Path to the CS2 demo/s file")
    parser.add_argument("--player", type=str, help="Player name to create video for (optional)")

    args = parser.parse_args()

    # Check if demo folder exists
    if not args.demo.exists():
        sys.exit(f"Demo file {args.demo} not found")
    demo_path = args.demo
    output_dir = "videos"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # If player name is provided, ensure it's a valid string
    if args.player:
        player_name = args.player.strip()
        if not player_name:
            sys.exit("Player name cannot be empty")
    else:
        player_name = None

    # Check if there is a config file
    config_path = Path('config.ini')
    if not config_path.exists():
        # Create a default config file if it doesn't exist
        with open(config_path, 'w') as config_file:
            config_file.write("[SETTINGS]\n")
            config_file.write("fps = 60\n")
            config_file.write("tickrate = 64\n")
            config_file.write("width = 800\n")
            config_file.write("height = 600\n")

    # Process the demo file and create videos for each player
    process_demo(demo_path, output_dir, player_name)

if __name__ == "__main__":
    main()