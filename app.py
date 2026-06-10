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

OBJECTIVE

Analyze the uploaded interior real estate image and perform MLS-quality professional color correction using object-aware segmentation, localized white balance correction, surface-based color normalization, and material-preserving adjustments.

The objective is to remove lighting-induced color contamination while preserving the original image quality, resolution, sharpness, texture, detail, and photographic realism.

SURFACE SEGMENTATION

Detect and generate accurate masks for:

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
* Artwork
* Architectural details

Analyze and process each surface independently before applying corrections.

WALL COLOR NORMALIZATION

Determine the true underlying paint color of every wall.

Separate actual paint color from lighting contamination caused by:

* Tungsten lighting
* Warm LED lighting
* Chandeliers
* Recessed lights
* Lamp spill
* Window reflections
* Skylights
* Daylight spill
* Sky reflections
* Mixed lighting conditions
* Reflected color from furniture and flooring
* Color bounce from nearby surfaces

Remove unwanted color contamination including:

* Yellow casts
* Orange casts
* Amber casts
* Blue casts
* Cyan casts
* Green casts
* Magenta casts
* Mixed-light contamination

Correct both warm and cool color contamination while preserving the true wall paint color.

Ensure wall color consistency throughout the room:

* No yellow corners
* No orange hotspots
* No amber discoloration
* No blue patches
* No cyan contamination
* No warm hotspots near fixtures
* No cool hotspots near windows
* No wall-to-wall color shifts
* No uneven wall tones
* No localized discoloration
* No lighting-induced color variation

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

TRIM, MOLDINGS, DOORS, AND FRAMES

Detect all trim, moldings, baseboards, doors, and frames.

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

CABINETS AND COUNTERTOPS

Detect cabinetry and countertop surfaces independently.

Correct only lighting contamination.

Preserve:

* Wood cabinet tones
* Painted cabinet colors
* Quartz colors
* Marble colors
* Granite colors
* Stone textures
* Material realism

Do not alter original material colors.

FLOORING CORRECTION

Detect flooring separately.

Preserve:

* Hardwood tones
* Natural wood warmth
* Tile colors
* Stone flooring colors
* Carpet colors

Remove only unwanted lighting contamination.

Do not neutralize naturally warm flooring materials.

FURNITURE, ARTWORK, AND DECOR PRESERVATION

Preserve authentic color characteristics of:

* Furniture
* Artwork
* Decorative objects
* Rugs
* Fabrics
* Pillows
* Curtains
* Accent pieces
* Wood finishes

Remove only artificial lighting contamination.

Maintain original material appearance and color identity.

WINDOW AND DAYLIGHT CONTROL

Detect windows, skylights, and daylight sources.

Remove blue and cyan contamination caused by:

* Daylight spill
* Skylight reflections
* Window reflections
* Exterior sky reflections

Prevent daylight contamination from affecting:

* Walls
* Ceilings
* Trim
* Doors
* Cabinets
* Furniture

LIGHT FIXTURE HANDLING

Detect all visible bulbs, lamps, chandeliers, sconces, pendants, recessed lights, and fixtures.

Allow bulbs and light sources to retain a realistic warm glow.

Prevent warm or cool light spill from contaminating:

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
* Countertops
* Furniture
* Flooring

Avoid global white balance adjustments.

Use surface-specific correction to achieve accurate colors while maintaining realism.

EXPOSURE AND TONAL BALANCING

Maintain natural brightness and contrast.

Preserve:

* Highlight detail
* Shadow detail
* Window detail
* Architectural detail
* Material texture

Avoid:

* Overexposure
* Excessive contrast
* Artificial HDR
* Unrealistic brightness
* Over-processing

IMAGE QUALITY PRESERVATION

Maintain the exact original photographic appearance.

DO NOT:

* Increase clarity
* Increase sharpness
* Increase texture
* Increase micro-contrast
* Increase local contrast
* Apply HDR processing
* Apply AI detail enhancement
* Apply edge enhancement
* Apply oversharpening
* Apply structure enhancement
* Apply artificial texture enhancement
* Apply aggressive denoising
* Create artificial detail

Preserve exactly:

* Original sharpness
* Original clarity
* Original texture
* Original noise/grain structure
* Original lens characteristics
* Original detail rendering
* Original depth perception
* Original photographic realism

Color correction must only modify:

* White balance
* Color balance
* Lighting contamination
* Surface color casts

No other visual enhancement should be applied.

RESOLUTION AND FILE QUALITY PRESERVATION

CRITICAL REQUIREMENT:

Maintain the exact original image dimensions and quality.

DO NOT:

* Resize the image
* Downscale the image
* Upscale the image
* Change aspect ratio
* Crop the image
* Compress the image
* Reduce image quality
* Re-encode using lower quality settings
* Generate a lower-resolution output

Preserve exactly:

* Original width
* Original height
* Original aspect ratio
* Original pixel count
* Original resolution
* Original image quality

The final output resolution must be identical to the uploaded image.

Example:

If input resolution is 3504×2336, output resolution must remain 3504×2336.

Perform all color corrections at full resolution.

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
* No warm halos remain
* No cool halos remain
* No localized color contamination remains
* No surface appears over-neutralized
* Original material colors are preserved
* Architectural details remain intact
* Lighting remains realistic
* Image sharpness matches the original
* Image clarity matches the original
* Texture intensity matches the original
* No additional detail has been created
* No HDR appearance is present
* No artificial enhancement is visible
* Output resolution equals input resolution
* Output dimensions equal input dimensions
* No resampling has occurred
* No compression artifacts have been introduced
* Color correction is the only visible modification

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
* No warm color contamination
* No cool color contamination
* No yellow, orange, amber, blue, cyan, green, or magenta casts
* No lighting-induced discoloration
* No increased sharpness
* No increased clarity
* No HDR effect
* No artificial enhancement
* No resolution loss
* No compression loss
* No change in image dimensions

FINAL GOAL

Produce a professional real estate image where walls display their true paint color, ceilings and trim appear clean neutral white, materials retain their authentic colors, all warm and cool lighting contamination has been removed, and the exact original image quality, resolution, sharpness, texture, detail level, and photographic realism are preserved.

Only color correction should be visible.

The final image must have the same dimensions, same resolution, same detail level, same sharpness, same texture, and same overall image quality as the original image while exhibiting professionally corrected and color-accurate surfaces.



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
                        
                        # Display the image
                        st.image(edited_image, caption='Corrected Image', use_container_width=True)
                        
                        # --- ADD DOWNLOAD BUTTON HERE ---
                        # Prepare the image for download
                        buf = io.BytesIO()
                        # Use PNG or JPEG. PNG is lossless and preserves quality perfectly.
                        edited_image.save(buf, format="PNG") 
                        byte_im = buf.getvalue()
                        
                        st.download_button(
                            label="Download Corrected Image",
                            data=byte_im,
                            file_name="corrected_real_estate.png",
                            mime="image/png"
                        )
                        # --------------------------------
                        
                        st.success("Correction applied!")
                        found = True
                        break
                
                if not found:
                    st.warning("Model response received, but no image found.")
                
                if not found:
                    st.warning("Model response received, but no image found.")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
