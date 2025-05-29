import OktaJwtVerifierModule from '@okta/jwt-verifier';
import { resourceServer } from "../config/index.js";

// Use the OktaJwtVerifier
const oktaJwtVerifier = new OktaJwtVerifierModule({
  clientId: resourceServer.oidc.clientId,
  issuer: resourceServer.oidc.issuer,
  assertClaims: resourceServer.assertClaims,
  testing: resourceServer.oidc.testing
});

const authMiddleware = (req, res, next) => {
  const authHeader = req.headers.authorization || '';
  const match = authHeader.match(/Bearer (.+)/);

  if (!match) {
    res.status(401);
    return next('Unauthorized');
  }

  const accessToken = match[1];
  const audience = resourceServer.assertClaims.aud;
  return oktaJwtVerifier.verifyAccessToken(accessToken, audience)
    .then((jwt) => {
      req.jwt = jwt;
      next();
    })
    .catch((err) => {
      res.status(401).send(err.message);
    });
}

export default authMiddleware;