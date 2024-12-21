# QR code generator

## Disclaimer

This is based on [https://github.com/gopro/labs/tree/master/docs/qr](https://github.com/gopro/labs/tree/master/docs/qr).

## Files

This directory provides two styles of QR code,

- [`qrtimecode.html`](qrtimecode.html) for GoPro compatible QR code, and
- [`qrtimecode+js.html`](qrtimecode+js.html) which bundles `qrcodeborder.js`, an external JavaScript module, into the single HTML file.

The `+js` version might be useful for local use (e.g., on an offline laptop, smartphone, tablet, etc.), because loading local JavaScript files may be blocked for security.

## Usage

A typical scenario would be as follows.

1. Open [`qrtimecode+js.html`](qrtimecode+js.html) on your smartphone $S$.
   - Be sure that $S$ shows a time-varying QR code at an appropriate framerate. If the framerate is too fast, every single exposure may include multiple QR code frames.
2. For each $C_i$ of your cameras $C_1$, $C_2$, ...
   1. Start recording.
   2. Show your smartphone $S$ to the camera $C_i$ for a few seconds.
   3. Leave $C_i$ keep capturing.
3. (do your main recording)
4. (optional) capture $S$ again.
5. Decode the QR codes in the captured videos and estimate the offset between them.

Open [../../ipynb/qrtimecode.ipynb](../../ipynb/qrtimecode.ipynb) for details.

