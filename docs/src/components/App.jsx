import "../styles/App.scss";
import HtmlImgComponent from "./HtmlImgContainer";
import PngImgContainer from "./PngImgContainer";
import PublicationComponent from "./publicationContainer";
import SelectContainer from "./SelectContainer";

function App() {
  return (
    <div className="App">
      <div className="titleContainer">
        <h1>CVNN vs RVNN PolSAR applications</h1>
      </div>

      <PublicationComponent></PublicationComponent>

      <SelectContainer></SelectContainer>

      <PngImgContainer></PngImgContainer>

      <HtmlImgComponent></HtmlImgComponent>
    </div>
  );
}

export default App;
