import os
import streamlit as st
import io
from PIL import Image as PILImage
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- 1. CLEAN ENVIRONMENT ---
# This prevents the SDK from auto-detecting Google Cloud/Vertex AI credentials
for var in ["GOOGLE_APPLICATION_CREDENTIALS", "GCP_PROJECT", "GOOGLE_CLOUD_PROJECT", "GOOGLE_GENAI_USE_VERTEXAI"]:
    os.environ.pop(var, None)

# --- 2. LOAD & VALIDATE ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("API Key not found. Please set GEMINI_API_KEY in your .env file.")
    st.stop()

# --- 3. INITIALIZE CLIENT ---
# By passing only api_key, we force the SDK to use the Gemini Developer API
client = genai.Client(api_key=api_key)

# --- 4. STREAMLIT UI ---
st.title("AI Real Estate White Balance Tool")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = PILImage.open(uploaded_file)
    image.thumbnail((1600, 1600))
    st.image(image, caption='Original', use_container_width=True)

    if st.button('Apply Correction'):
        with st.spinner('Refining lighting...'):
            try:
                prompt = (
                     '''ROLE: Professional Real Estate Interior Color Correction Engine

OBJECTIVE:

Analyze the uploaded interior real estate image and perform MLS-quality professional color correction using object-aware segmentation, localized white balance correction, surface-based color normalization, and material-preserving adjustments.

SURFACE SEGMENTATION

Detect and generate independent masks for:

* Walls
* Ceilings
* Trim and moldings
* Doors and door frames
* Cabinets
* Countertops
* Flooring
* Furniture
* Light fixtures
* Windows
* Decorative objects
* Architectural details

Analyze and process each surface independently before applying color correction.

WALL COLOR NORMALIZATION

Determine the true underlying paint color of every wall.

Separate actual paint color from lighting contamination caused by:

* Tungsten lighting
* Warm LED lighting
* Chandeliers
* Recessed lights
* Lamp spill
* Window reflections
* Sky reflections
* Mixed lighting conditions
* Color bounce from nearby surfaces

Remove unwanted lighting-induced color contamination including:

* Yellow casts
* Orange casts
* Amber casts
* Blue casts
* Cyan casts
* Green casts
* Magenta casts
* Mixed-light contamination

Correct both warm and cool color contamination while preserving the true paint color.

Maintain wall color consistency throughout the room:

* No yellow corners
* No orange hotspots
* No amber discoloration
* No blue patches
* No cyan contamination
* No warm hotspots near fixtures
* No cool hotspots near windows
* No wall-to-wall color shifts
* No uneven wall tones
* No lighting-induced discoloration

Walls should appear clean, uniform, natural, and professionally painted while retaining their original color.

CEILING CORRECTION

Detect all ceiling surfaces separately from walls.

For white or off-white ceilings:

* Restore neutral white appearance
* Remove yellow contamination
* Remove orange contamination
* Remove amber contamination
* Remove blue contamination
* Remove cyan contamination
* Remove green contamination
* Remove magenta contamination
* Remove color bleeding from fixtures
* Remove daylight contamination
* Remove sky-color contamination
* Remove lamp spill
* Eliminate warm halos around fixtures
* Eliminate cool halos around windows and skylights

Maintain:

* Ceiling texture
* Surface detail
* Architectural features
* Crown molding definition
* Natural shadows

Ceilings must appear consistently neutral white throughout the image.

TRIM, MOLDINGS, AND DOORS

Detect white trim, moldings, baseboards, doors, and door frames.

Remove all lighting-induced color contamination.

Eliminate:

* Yellow casts
* Orange casts
* Amber casts
* Blue casts
* Cyan casts
* Green casts
* Magenta casts

Restore a clean neutral-white appearance while preserving:

* Surface texture
* Material characteristics
* Gloss and reflections
* Natural shadow detail

MATERIAL PRESERVATION

Preserve the authentic color characteristics of:

* Hardwood flooring
* Wood cabinetry
* Furniture
* Stone countertops
* Quartz surfaces
* Marble surfaces
* Artwork
* Decorative accents
* Fabrics
* Rugs
* Natural wood finishes

Do not neutralize materials that are naturally warm or cool in color.

Only remove unwanted lighting contamination.

Preserve the original appearance and material identity of every object.

LIGHT FIXTURE HANDLING

Detect all visible bulbs, lamps, chandeliers, sconces, pendants, and light fixtures.

Allow light sources to retain a realistic warm glow.

However, prevent warm or cool color contamination from affecting:

* Walls
* Ceilings
* Trim
* Doors
* Cabinets
* Furniture
* Flooring
* Countertops

Light sources may remain warm and realistic, but surrounding surfaces must retain their true material color.

LOCALIZED WHITE BALANCE CORRECTION

Apply object-aware localized white balance correction.

Independently correct:

* Walls
* Ceilings
* Trim
* Doors
* Cabinets
* Furniture
* Floors
* Countertops

Avoid global white balance adjustments that uniformly affect the entire image.

Use surface-specific correction to achieve accurate colors while maintaining a natural appearance.

EXPOSURE AND TONAL BALANCING

Maintain natural brightness and contrast.

Improve overall image clarity while preserving:

* Highlight detail
* Shadow detail
* Window detail
* Architectural detail
* Material texture

Avoid:

* Overexposure
* Flat appearance
* Artificial HDR effects
* Excessive contrast
* Over-processing

QUALITY CONTROL

Verify that:

* White ceilings appear neutral white
* White trim appears neutral white
* White doors appear neutral white
* Wall color is consistent throughout the room
* No yellow casts remain
* No orange casts remain
* No amber casts remain
* No blue casts remain
* No cyan casts remain
* No green casts remain
* No magenta casts remain
* No warm halos remain around fixtures
* No cool halos remain around windows
* No localized color contamination remains
* No surface appears over-neutralized
* Original material colors are preserved
* Architectural details remain intact
* Lighting remains realistic and natural
* The image appears professionally color-corrected

OUTPUT REQUIREMENTS

Generate a bright, clean, realistic, MLS-quality real estate photograph with:

* Uniform wall color
* Neutral white ceilings
* Neutral white trim
* Neutral white doors
* Accurate material colors
* Balanced white balance
* Natural contrast
* Realistic lighting
* Consistent color throughout the room
* No visible warm color contamination
* No visible cool color contamination
* No yellow, orange, amber, blue, cyan, green, or magenta casts
* No lighting-induced discoloration
* No artificial or over-processed appearance

FINAL GOAL

Produce a professional real estate image where walls display their true paint color, ceilings and trim appear clean neutral white, materials retain their authentic colors, and all warm or cool lighting contamination has been removed while maintaining a natural, realistic, MLS-ready appearance.

 '''
                )
                
                # API Call
                response = client.models.generate_content(
                    model='gemini-3.1-flash-image', 
                    contents=[prompt, image],
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"]
                    )
                )
                
                # Display output
                found = False
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        edited_image = PILImage.open(io.BytesIO(part.inline_data.data))
                        st.image(edited_image, caption='Corrected Image', use_container_width=True)
                        st.success("Correction applied!")
                        found = True
                        break
                
                if not found:
                    st.warning("Model response received, but no image found.")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
