<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->

<a name="readme-top"></a>

<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/DickyAdi/batagor-recognition">
  </a>

<h3 align="center">Batagor Recognition</h3>

  <p align="center">
    An unsupervised-model using CNN-SVM to recognize if an image is a Batagor (Indonesian dishes)
    <br />
    <a href="https://github.com/DickyAdi/
batagor-recognition"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/DickyAdi/batagor-recognition">View Demo</a>
    ·
    <a href="https://github.com/DickyAdi/batagor-recognition/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/DickyAdi/batagor-recognition/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

This is a test project i can't finished because i cannot think any of the solution within the time limit. However, i finally came up with a solution in combining Pre-trained Convolutional Neural Network (ResNet-18) with One Class SVM. The main idea of this approach is by comparing the new images/instances to the extracted features from the dataset. The CNN is used for extracting the features from the image and SVM act as a predictor (outlier/anomaly detection).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

To run this project, follow these prerequisites and instruction below.

### Prerequisites

Install all the requirements by running this command below

- requirements.txt
  ```sh
  pip install -r requirements.txt
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/DickyAdi/batagor-recognition.git
   ```
2. Install NPM packages as above
   ```sh
   pip install -r requirements.txt
   ```
3. Run the streamlit apps by running this command below
   ```sh
   streamlit run app.py
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

To use this apps you can just simply upload your images to the given file uploader. After a while, the results will be written on your screen.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

Dicky Adi Naufal Farhansyah - dickyadi44@gmail.com

Project Link: [https://github.com/DickyAdi/batagor-recognition](https://github.com/DickyAdi/batagor-recognition)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/DickyAdi/batagor-recognition.svg?style=for-the-badge
[contributors-url]: https://github.com/DickyAdi/batagor-recognition/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/DickyAdi/batagor-recognition.svg?style=for-the-badge
[forks-url]: https://github.com/DickyAdi/batagor-recognition/network/members
[stars-shield]: https://img.shields.io/github/stars/DickyAdi/batagor-recognition.svg?style=for-the-badge
[stars-url]: https://github.com/DickyAdi/batagor-recognition/stargazers
[issues-shield]: https://img.shields.io/github/issues/DickyAdi/batagor-recognition.svg?style=for-the-badge
[issues-url]: https://github.com/DickyAdi/batagor-recognition/issues
[license-shield]: https://img.shields.io/github/license/DickyAdi/batagor-recognition.svg?style=for-the-badge
[license-url]: https://github.com/DickyAdi/batagor-recognition/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/dickyadi
