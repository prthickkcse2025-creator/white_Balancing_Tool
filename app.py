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

SURFACE SEGMENTATION:

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

Process each surface independently.

WALL COLOR NORMALIZATION:

Determine the true underlying paint color of every wall.

Separate paint color from lighting contamination caused by:

* Tungsten lighting
* Warm LEDs
* Chandeliers
* Recessed lights
* Lamp spill
* Window reflections
* Mixed lighting conditions
* Color bounce from surrounding materials

Remove unwanted:

* Yellow casts
* Orange casts
* Amber casts
* Green casts
* Magenta casts
* Mixed-light contamination

Preserve the original wall paint color.

Ensure wall color consistency across all wall surfaces:

* No yellow corners
* No orange hotspots
* No warm patches
* No uneven wall tones
* No wall-to-wall color shifts
* No lighting-induced discoloration

Walls should appear clean, uniform, and naturally painted while retaining original color.

CEILING CORRECTION:

Detect all ceiling surfaces separately from walls.

For white or off-white ceilings:

* Restore neutral white appearance
* Remove yellow contamination
* Remove orange contamination
* Remove amber contamination
* Remove green contamination
* Remove magenta contamination
* Remove color bleeding from fixtures
* Remove lamp spill
* Eliminate warm halos around light sources

Maintain:

* Ceiling texture
* Shadows
* Architectural detail
* Crown molding definition

Ceilings must appear consistently neutral and naturally white.

TRIM, DOORS, AND MOLDINGS:

Detect white trim, moldings, and doors.

Remove all lighting-induced color contamination.

Restore neutral white appearance while preserving:

* Surface texture
* Gloss
* Material detail
* Shadow definition

MATERIAL PRESERVATION:

Preserve the natural color characteristics of:

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

Do not neutralize naturally warm materials.

Only remove artificial lighting contamination.

LIGHT FIXTURE HANDLING:

Detect visible light fixtures and bulbs.

Allow bulbs and light sources to retain realistic warm illumination.

Prevent warm color spill from affecting:

* Walls
* Ceilings
* Trim
* Doors
* Cabinets
* Furniture
* Flooring

Light sources may remain warm, but surrounding surfaces must retain true material color.

LOCALIZED WHITE BALANCE:

Apply object-based localized white balance correction.

Independently correct:

* Walls
* Ceilings
* Trim
* Doors
* Cabinets
* Furniture
* Floors

Avoid global white balance adjustments that affect the entire image uniformly.

QUALITY CONTROL:

Verify that:

* White ceilings appear neutral white
* White trim appears neutral white
* White doors appear neutral white
* Wall color is uniform throughout the room
* No yellow halos remain
* No orange halos remain
* No green contamination remains
* No magenta contamination remains
* No localized color casts remain
* No over-neutralization occurs
* Material colors remain authentic
* Architectural details remain intact
* Lighting remains realistic
* Image remains natural and believable

OUTPUT REQUIREMENTS:

Generate a bright, clean, realistic, MLS-quality real estate photograph with:

* Uniform wall color
* Neutral white ceilings
* Neutral white trim
* Accurate material colors
* Natural contrast
* Balanced exposure
* Realistic lighting
* Professional real estate photography appearance
* No visible lighting-induced color contamination
* No artificial or over-processed appearance
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