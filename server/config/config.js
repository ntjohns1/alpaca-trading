// config.js

import { resolve } from 'path';
import { parse } from 'dotenv';
import { existsSync, readFileSync } from 'fs';

const ISSUER = process.env.ISSUER || 'https://{yourOktaDomain}.com/oauth2/default';
const SPA_CLIENT_ID = process.env.CLIENT_ID || '{spaClientId}';
const OKTA_TESTING_DISABLEHTTPSCHECK = process.env.OKTA_TESTING_DISABLEHTTPSCHECK ? true : false;

const resourceServer = {
  port: 8000,
  oidc: {
    clientId: SPA_CLIENT_ID,
    issuer: ISSUER,
    testing: {
      disableHttpsCheck: OKTA_TESTING_DISABLEHTTPSCHECK
    }
  },
  assertClaims: {
    aud: 'api://default',
    cid: SPA_CLIENT_ID
  }
};

export default resourceServer;

