# Auto Video System

The system that runs the YoungSponge (@youngspongeshoes) short-video channels. It writes, voices, edits and posts ~15 short videos a day across Instagram, YouTube, Facebook, Threads and TikTok, then pulls the numbers back to decide what to make next.

## How it works
1. **Script** - a script is written for each video.
2. **Voice** - the script is read in a trained AI clone of my own voice.
3. **Edit** (`edit/`) - `engine.py` takes a reference reel plus my footage and voiceover and rebuilds it: picks the best takes, matches b-roll to each spoken line, adds captions, transitions and sound effects, then checks the result (caption safe-zones, no clip reused, no frozen frames) before writing the file.
4. **Footage** - six cameras in the basement record on motion; a nightly job throws out dead/duplicate clips, cuts the good moments and tags them so the editor can find them.
5. **Post** (`post/`) - `dispatch.py` uploads each finished video to every platform through their official APIs and records what went where.
6. **Learn** - views, reach and watch time are pulled back daily and fed into the next day's scripts.

`ai-video/` holds the AI video generation and frame-checking tools.

All passwords and tokens live outside the code in local files and are not in this repo.
