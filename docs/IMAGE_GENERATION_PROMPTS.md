# Sign Image Generation — Prompt Kit

Prompts for generating reference images (Gemini / ChatGPT / Midjourney) for the
Learning-mode sign display.

> ## ⚠️ Read this before generating anything
>
> **Hands are the single worst failure case for image generators** — extra
> fingers, fused fingers, wrong counts. For a sign-language teaching app that
> is not cosmetic: a wrong handshape teaches someone the wrong sign.
>
> **Verify every single image against a reference before using it** (sources in
> §5). Regenerate anything that isn't exactly right. Budget for roughly half
> the outputs being unusable.
>
> This is also why the template below includes an explicit anatomical
> description — asking for "ASL letter M" alone reliably produces the wrong
> hand.

---

## 1. Master prompt template

Replace `{SIGN_LABEL}` and `{HANDSHAPE_DESCRIPTION}` from the tables in §3/§4.

```
A photorealistic close-up photograph of a single human right hand performing
the American Sign Language (ASL) sign for "{SIGN_LABEL}".

HAND POSITION (follow this exactly — this is the most important part):
{HANDSHAPE_DESCRIPTION}

ANATOMY REQUIREMENTS:
- Exactly five fingers, anatomically correct, clearly separated and countable
- Natural proportions, realistic knuckles and joints
- Clean, short natural nails — no nail polish
- No jewellery, no watch, no rings, no tattoos
- Bare forearm entering from the bottom edge of the frame, no sleeve visible

COMPOSITION:
- Single hand only, centred, filling roughly 70% of the frame
- Forearm vertical, rising from the bottom edge
- Square 1:1 aspect ratio, high resolution
- Plain solid warm cream / ivory background (#FDF6E3), completely uniform,
  no texture, no gradient, no props, no background objects

LIGHTING & STYLE:
- Soft, even studio lighting from the front-left
- Gentle natural shadow beneath the hand for depth
- Photorealistic, sharp focus, neutral medium skin tone
- Clean commercial reference-photograph aesthetic, like an educational chart

TEXT LABEL:
- Place the text "{SIGN_LABEL}" in the TOP RIGHT corner
- Bold black sans-serif, clean and legible, small (about 8% of image height)
- The text must be spelled exactly as written, with no other text anywhere
  in the image
```

### Negative prompt (for tools that support one)

```
extra fingers, missing fingers, six fingers, fused fingers, deformed hand,
mangled hand, distorted anatomy, two hands, multiple hands, blurry, low
quality, cartoon, illustration, drawing, 3d render, cluttered background,
patterned background, jewellery, watch, long fingernails, nail polish,
watermark, extra text, misspelled text, gibberish text
```

---

## 2. How to use

1. Pick a row from §3 (letters) or §4 (words).
2. Paste the template, substituting `{SIGN_LABEL}` and `{HANDSHAPE_DESCRIPTION}`.
3. **Check the result against a reference in §5.** Regenerate if wrong.
4. Save as `{Label}.jpg` into `frontend/dristi/src/static/`
   (e.g. `E.jpg`, `Hello.jpg`) to match the existing `A.jpg`–`D.jpg`.

**Note on consistency:** your existing `A.jpg`–`D.jpg` put the letter in the
**bottom-right**. The template above says top-right as requested — if you want
the new images to match the four you already have, change `TOP RIGHT` to
`BOTTOM RIGHT` in the template. Pick one and stay with it.

You already have **A, B, C, D** — you only need to generate the rest.

---

## 3. Letters

`{SIGN_LABEL}` = the letter. Palm faces the viewer unless stated otherwise.

**J and Z are deliberately excluded** — both are *motion* letters (they trace a
shape in the air) and cannot be represented by a static image.

| Label | `{HANDSHAPE_DESCRIPTION}` |
|---|---|
| A | Closed fist, palm facing the viewer. Thumb straight and resting flat along the *side* of the index finger — not tucked inside the fist, not across the front. |
| B | All four fingers fully extended, straight and pressed tightly together, pointing up. Thumb folded flat across the middle of the palm. Palm faces the viewer. |
| C | Fingers held together and curved, thumb curved below them, forming a clear letter "C" shape. Palm faces sideways to the left, so the C opening faces the viewer. |
| D | Index finger extended straight up. Middle, ring and little fingers curled down so their tips touch the tip of the thumb, forming a round circle below the index. |
| E | All four fingers bent down at the knuckles so the fingertips rest on the top of the thumb. Thumb tucked horizontally beneath the curled fingers. Compact, closed shape. |
| F | Tip of the index finger and tip of the thumb pressed together forming a circle. Middle, ring and little fingers extended straight up and slightly apart. |
| G | Index finger extended and pointing sideways (to the left), thumb extended parallel above it, a small gap between them. Remaining three fingers curled into the palm. |
| H | Index and middle fingers extended straight and held tightly together, pointing sideways to the left. Ring and little fingers curled down, thumb resting over them. |
| I | Little finger (pinky) extended straight up. Index, middle and ring fingers curled into a fist, thumb resting across their front. |
| K | Index finger extended straight up, middle finger extended upward and angled away from it, thumb pressed against the base of the middle finger between the two. Ring and little fingers curled down. |
| L | Index finger extended straight up, thumb extended straight out horizontally, forming a clean right-angle "L". Middle, ring and little fingers curled into the palm. |
| M | Thumb folded across the palm, with the index, middle AND ring fingers folded down over the top of the thumb. Little finger curled beside them. Three fingers over the thumb. |
| N | Thumb folded across the palm, with the index AND middle fingers folded down over the top of the thumb. Ring and little fingers curled beside them. Two fingers over the thumb. |
| O | All four fingers curved down and the thumb curved up so all fingertips meet the thumb tip, forming a rounded hollow letter "O". Palm faces the viewer. |
| P | Index finger extended, middle finger extended and angled away, thumb between them at the base of the middle finger — the whole hand rotated so the fingers point DOWNWARD. |
| Q | Index finger and thumb extended parallel with a small gap, both pointing DOWNWARD toward the floor. Remaining three fingers curled into the palm. |
| R | Index and middle fingers extended upward and crossed over one another, index in front. Ring and little fingers curled down, thumb resting across them. |
| S | Closed fist with the thumb crossing horizontally in FRONT of the folded fingers, over the middle of them. Palm faces the viewer. |
| T | Closed fist with the thumb inserted between the index and middle fingers, thumb tip poking out slightly. Palm faces the viewer. |
| U | Index and middle fingers extended straight up and held tightly together, touching along their length. Ring and little fingers curled down, thumb across them. |
| V | Index and middle fingers extended straight up and spread clearly apart in a "V". Ring and little fingers curled down, thumb resting across them. |
| W | Index, middle and ring fingers extended straight up and spread apart. Thumb and little finger tips pressed together across the palm. Exactly three fingers up. |
| X | Index finger raised and bent into a hook or claw at the middle joint. All other fingers curled into a fist, thumb resting against the side. |
| Y | Thumb and little finger both extended outward in opposite directions. Index, middle and ring fingers curled into the palm. The "hang loose" shape. |

---

## 4. Words

⚠️ **Read this first.** ASL word signs are **movement plus body location** —
"Hello" is a hand travelling outward from the temple. A static image can only
show one representative moment, so these are inherently weaker than the letter
images. Two ways to handle it:

- **Add a motion arrow.** Append this to the template:
  > `Include a subtle curved grey arrow showing the direction the hand moves, starting at the hand and following the described path.`
- **Better: use short video clips for words** and reserve generation for
  letters. Word signs are what `record_word_clips.py` and the WLASL clips are
  for.

For these, also change the composition line to allow the head/shoulder when the
sign's *location* matters (Hello, Thankyou, Sorry, Name) — otherwise the sign is
ambiguous:

> `Show the person's head and upper shoulder in frame so the position of the hand relative to the face is clear.`

| Label | `{HANDSHAPE_DESCRIPTION}` |
|---|---|
| Hello | Flat hand, fingers extended and together, thumb alongside. Held at the side of the forehead near the temple, then moving outward and away from the head in an arc, like a relaxed salute. |
| Bye | Flat open hand raised at shoulder height, palm facing the viewer, fingers extended. Fingers bending down and up repeatedly — a wave. |
| Please | Flat open hand, fingers together, palm pressed flat against the centre of the chest, moving in a circular motion. |
| Thankyou | Flat hand, fingers extended and together, fingertips touching the chin, then moving forward and down away from the face toward the viewer. |
| Sorry | Hand closed into a fist with the thumb alongside, pressed against the centre of the chest, moving in a circular motion. |
| Yes | Hand closed into a fist, palm facing the viewer, bending up and down at the wrist — like a head nodding. |
| No | Index and middle fingers extended together, tapping down onto the extended thumb, closing like a beak snapping shut. |
| Help | One flat open palm facing up, with the other hand closed into a fist with thumb raised resting on top of it, both lifting upward together. |
| Good | Flat hand, fingers together, fingertips touching the chin, then moving forward and down to land palm-up on the other open palm. |
| Love | Both arms crossed over the chest, hands closed into fists, in a self-hug. |
| Friend | Both index fingers curved into hooks, linking together, then unlinking and re-linking the opposite way. |
| Eat | Fingers and thumb pinched together as if holding food, tapping the fingertips against the lips. |
| Drink | Hand curved into a "C" shape as if holding a cup, tilting up toward the mouth. |
| Water | Index, middle and ring fingers extended and spread (the "W" shape), tapping the index finger against the chin. |
| Home | Fingers and thumb pinched together, touching first the corner of the mouth then the cheek. |
| Work | Both hands closed into fists, one tapping down on the wrist of the other repeatedly. |
| More | Both hands with all fingertips and thumb pinched together into flattened "O" shapes, tapping the fingertips of both hands together. |
| Stop | One flat hand held vertically, fingers extended and together, chopping down sharply onto the flat open palm of the other hand. |
| Name | Index and middle fingers extended together on both hands, one pair tapping across the top of the other in an X. |
| You | Index finger extended, pointing directly forward at the viewer. All other fingers curled into the palm. |
| Me | Index finger extended, pointing back at the signer's own chest. All other fingers curled into the palm. |

---

## 5. Reference sources — check every image against these

Give these to the model in the prompt ("match the reference on Lifeprint"), but
more importantly **use them yourself to verify output**:

- **Lifeprint / ASL University** — https://www.lifeprint.com/asl101/fingerspelling/
  (Dr. Bill Vicars; the most widely used free ASL reference)
- **Handspeak** — https://www.handspeak.com/spell/ (clear alphabet + word dictionary)
- **Signing Savvy** — https://www.signingsavvy.com/ (video per sign — best for the word signs)
- **Gallaudet University** — https://gallaudet.edu/ (authoritative Deaf-education source)

For the **word** signs especially, use Signing Savvy or Handspeak video — a
static reference can't tell you whether your generated pose is the right moment
of the movement.

---

## 6. Suggested workflow

1. Generate letters first — they're static, so generation actually works well.
2. Verify each against Lifeprint. Discard and regenerate failures.
3. For words, prefer **clips** over generated stills. If you do generate them,
   add the motion arrow and show the head when location matters.
4. Save into `frontend/dristi/src/static/` as `{Label}.jpg`.
5. In your report, state plainly that reference images were AI-generated and
   individually verified against an authoritative ASL source. That's a
   defensible methodology; unverified generated images are not.
