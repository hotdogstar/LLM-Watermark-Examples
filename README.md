# LLM Watermarking Examples in Python

Basic examples and implementations of several Large Language Model (LLM) watermarking techniques in Python. These implementations are designed to be simple and self-contained, requiring no external libraries beyond standard Python libraries like `numpy` and `hashlib`, and **do not require access to actual LLMs**. This makes them useful for educational purposes and for understanding how watermarking techniques can be applied in practice. This repository includes implementations of the following watermarking schemes:

## Implemented Watermarks

- **EXP Watermark**
- **KGW Watermark (Hard Red List)**
- **KGW Watermark (Soft Red List)**

## Understanding the Results

- **Z-Score Interpretation:**
  - A **z-score > 4** indicates that the text is likely watermarked.
  - A **z-score ≈ 0** suggests that the text is not watermarked.
  (The value of 4 for rejecting the null hypothesis has been chosen arbitrarily and a different threshold can be used)

## Further Information

For more details on the watermarking schemes implemented in this repository, refer to the following papers and resources:

- **EXP Watermark:**
  - *Title:* "A Watermark for Large Language Models"
  - *Authors:* Scott Aaronson
  - *Link:* [ Watermarking of Large Language Models ](https://www.youtube.com/watch?v=2Kx9jbSMZqA&t=2258s)

- **KGW Watermark (Hard Green List) & Soft Red List Watermark:**
  - *Title:* "A Watermark for Large Language Models"
  - *Authors:* John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, Tom Goldstein
  - *Link:* [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)

## TODO

- Implement additional watermarking schemes
- Evaluate the robustness of the watermarks under various transformations.

## License

This project is licensed under the [MIT License](LICENSE).

