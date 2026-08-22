'use strict'

/**
 * The only bridge between the dashboard and the desktop shell.
 *
 * Deliberately narrow: the renderer can ask whether a key is set and can set
 * one, but can never read it back. A stored secret that the page can read is
 * a secret that any injected script can exfiltrate.
 */

const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('auditorDesktop', {
  isDesktop: true,

  /** Which port the backend came up on this launch. */
  apiBase: () => ipcRenderer.invoke('auditor:api-base'),

  /** `{ set, canPersist }` -- never the key itself. */
  getApiKeyStatus: () => ipcRenderer.invoke('auditor:get-api-key'),

  /** Store the key and apply it to the running backend. Empty string clears. */
  setApiKey: (key) => ipcRenderer.invoke('auditor:set-api-key', key),

  /** Fired by the Settings menu item. */
  onOpenSettings: (handler) => {
    const listener = () => handler()
    ipcRenderer.on('auditor:open-settings', listener)
    return () => ipcRenderer.removeListener('auditor:open-settings', listener)
  },
})
